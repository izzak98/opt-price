import torch
from torch import nn
from torch.nn import functional as F
from accelerate import Accelerator
from q_storm.storm_utils.varrho import calc_varrho
from q_storm.qStorm import QStorm

accelerator = Accelerator()


class StormTrainer():
    def __init__(self,
                 lambda_p=1,
                 lambda_u=1,
                 lambda_o=1,
                 lambda_i=1,):
        self.lambda_p = lambda_p
        self.lambda_u = lambda_u
        self.lambda_o = lambda_o
        self.lambda_i = lambda_i
        assert all(l >= 1 for l in [lambda_p, lambda_u, lambda_o, lambda_i])

    def residual_loss(self,
                      model: torch.nn.Module,
                      S_prime: torch.Tensor,    # Normalized stock price S' = S/K
                      t_prime: torch.Tensor,    # Normalized time-to-maturity τ = T - t
                      rf: torch.Tensor,         # Risk-free rate
                      varpi_q: torch.Tensor,    # Additional input (context-dependent)
                      varpi_grid: torch.Tensor,  # Grid data for varpi
                      varpi_pdf: torch.Tensor,  # PDF for varpi
                      w_t: torch.Tensor,        # Brownian motion W_t
                      var_rho_taus: torch.Tensor,  # Pre-computed data for varrho(τ)
                      ) -> torch.Tensor:
        """
        Compute the residual loss for the normalized SPDE.
        """
        # Forward pass: predict normalized option price V' (or U) in normalized space
        V_prime = model(S_prime, t_prime, rf, varpi_q)  # Shape: (batch_size, 1)

        varrho_t = t_prime  # * 365  # Convert τ to days
        # Compute the additional normalized parameter varrho(tau) based on w_t
        w_t = w_t * varrho_t  # Normalize w_t by time
        varrho = calc_varrho(var_rho_taus, varpi_q, varpi_grid,
                             varpi_pdf, w_t, varrho_t)  # varrho(τ)

        # Compute derivatives of V' w.r.t. the inputs
        # Use automatic differentiation for PDE residual terms
        grads = torch.autograd.grad(
            outputs=V_prime,
            inputs=(S_prime, t_prime),  # Differentiate w.r.t. S' and τ
            grad_outputs=torch.ones_like(V_prime),
            create_graph=True,
            retain_graph=True
        )
        V_prime_S = grads[0]         # ∂V'/∂S' (Shape: batch_size x 1)
        V_prime_tau = grads[1]       # ∂V'/∂τ  (Shape: batch_size x 1)

        # Compute second derivative w.r.t. S' (∂²V'/∂S'²)
        V_prime_S2 = torch.autograd.grad(
            outputs=V_prime_S,
            inputs=S_prime,
            grad_outputs=torch.ones_like(V_prime_S),
            create_graph=True,
            retain_graph=True
        )[0]  # Shape: batch_size x 1

        # Residual terms in the normalized SPDE
        # First term: ∂U/∂τ
        term1 = V_prime_tau

        # Second term: - (1/2) (S')^2 varrho^2 ∂²U/∂(S')²
        term2 = 0.5 * (S_prime ** 2) * (varrho ** 2) * V_prime_S2

        # Fourth term: + rf S' ∂U/∂S'
        term3 = rf * S_prime * V_prime_S

        # Fifth term: - rf U
        term4 = - rf * V_prime
        residual = term1 + term2 + term3 + term4
        # Residual loss is the mean squared residual of the SPDE
        residual_loss = torch.mean(residual ** 2)
        return residual_loss

    def payoff_loss(self, model, S_prime, t_max, rf, varpi_q):
        raise NotImplementedError("payoff_loss method must be implemented.")

    def under_loss(self, model, S_min, t_prime, rf, varpi_q):
        raise NotImplementedError("under_loss method must be implemented.")

    def over_loss(self, model, S_max, t_prime, rf, varpi_q):
        raise NotImplementedError("over_loss method must be implemented.")

    def inequality_loss(self, model, S_prime, t_prime, rf, varpi_q):
        raise NotImplementedError("inequality_loss method must be implemented.")

    def forward(self, model, sampled_data, taus, mc_samples):
        total_residual_loss = 0
        S_prime = sampled_data["S_prime"]
        t_prime = sampled_data["t_prime"]
        rf = sampled_data["rf"]
        varpi_q = sampled_data["varpi_q"]
        varpi_grid = sampled_data["varpi_grids"]
        varpi_pdf = sampled_data["varpi_pdfs"]
        S_min = torch.zeros_like(S_prime).to(accelerator.device)

        S_max = torch.ones_like(S_prime).to(accelerator.device)*5

        W_t = torch.randn(mc_samples, S_prime.size(0), 1).to(accelerator.device)
        for w_t in W_t:
            residual_loss = self.residual_loss(
                model=model,
                S_prime=S_prime,
                t_prime=t_prime,
                rf=rf,
                varpi_q=varpi_q,
                varpi_grid=varpi_grid,
                varpi_pdf=varpi_pdf,
                w_t=w_t,
                var_rho_taus=taus
            )
            total_residual_loss += residual_loss
        total_residual_loss = total_residual_loss / mc_samples

        t_max = torch.zeros_like(t_prime).to(accelerator.device)
        payoff_loss = self.payoff_loss(model, S_prime, t_max, rf, varpi_q)

        under_loss = self.under_loss(model, S_min, t_prime, rf, varpi_q)

        over_loss = self.over_loss(model, S_max, t_prime, rf, varpi_q)

        inequality_loss = self.inequality_loss(model, S_prime, t_prime, rf, varpi_q)

        total_loss = (self.lambda_p * payoff_loss +
                      self.lambda_u * under_loss +
                      self.lambda_o * over_loss +
                      self.lambda_i * inequality_loss +
                      total_residual_loss)

        unadjusted_loss = (payoff_loss + under_loss + over_loss +
                           inequality_loss + total_residual_loss)
        losses = {
            "total_loss": total_loss,
            "unadjusted_loss": unadjusted_loss,
            "payoff_loss": payoff_loss,
            "under_loss": under_loss,
            "over_loss": over_loss,
            "inequality_loss": inequality_loss,
            "residual_loss": total_residual_loss
        }
        return losses


class CallStormTrainer(StormTrainer):
    def payoff_loss(self, model, S_prime, t_max, rf, varpi_q):
        assert torch.all(t_max == 0)
        payoff = torch.relu(S_prime - 1.0)
        V_prime = model(S_prime, t_max, rf, varpi_q)
        loss = torch.mean((V_prime - payoff)**2)
        return loss

    def under_loss(self, model, S_min, t_prime, rf, varpi_q):
        V_prime = model(S_min, t_prime, rf, varpi_q)
        return torch.mean((V_prime)**2)

    def over_loss(self, model, S_max, t_prime, rf, varpi_q):
        V_prime = model(S_max, t_prime, rf, varpi_q)
        payout = S_max - 1.0  # Corrected from exp(-rf * t_prime)
        loss = torch.mean((V_prime - payout)**2)
        return loss

    def inequality_loss(self, model, S_prime, t_prime, rf, varpi_q):
        """
        Enforces the American call constraint: V'(S',τ) >= max(S' - 1,0).
        Any violation => penalty in the loss.
        """
        # Model prediction
        V_prime = model(S_prime, t_prime, rf, varpi_q)
        # Immediate exercise value in normalized space
        exercise_value = F.relu(S_prime - 1.0)

        # The difference (exercise_value - V_prime) should be <= 0
        # If (exercise_value - V_prime) > 0 => violation
        violation = exercise_value - V_prime

        # We penalize only positive violations (i.e., clamp below at zero)
        penalty = F.relu(violation)
        # Typically squared penalty:
        return torch.mean(penalty ** 2)
