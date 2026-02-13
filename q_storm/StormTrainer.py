import torch
from torch import nn
from torch.nn import functional as F
from accelerate import Accelerator
from q_storm.storm_utils.varrho import calc_varrho
from q_storm.qStorm import QStorm

accelerator = Accelerator()


class StormTrainer():
    """
    Trainer for qStorm model that enforces SPDE constraints via physics-informed loss.

    The total loss is:
    L_total = λ_p * L_payoff + λ_u * L_under + λ_o * L_over + λ_i * L_inequality + L_residual

    Where:
    - L_residual: Monte Carlo averaged squared PDE residual
    - L_payoff: Terminal condition (at t'=0)
    - L_under: Lower boundary (at S'=0)
    - L_over: Upper boundary (at S'=S_max)
    - L_inequality: American option constraint violation penalty
    """

    def __init__(self,
                 lambda_p=1,
                 lambda_u=1,
                 lambda_o=1,
                 lambda_i=1,):
        """
        Initialize loss weights. All weights must be >= 1.

        Args:
            lambda_p: Weight for terminal payoff boundary condition
            lambda_u: Weight for lower boundary condition (S'=0)
            lambda_o: Weight for upper boundary condition (S'=S_max)
            lambda_i: Weight for American option inequality constraint
        """
        self.lambda_p = lambda_p
        self.lambda_u = lambda_u
        self.lambda_o = lambda_o
        self.lambda_i = lambda_i
        assert all(l >= 1 for l in [lambda_p, lambda_u, lambda_o, lambda_i])

    def residual_loss(self,
                      model: torch.nn.Module,
                      # Normalized stock price S' = S/K, shape: (batch_size, 1)
                      S_prime: torch.Tensor,
                      # Time-to-expiration in DAYS (not normalized), shape: (batch_size, 1)
                      t_prime: torch.Tensor,
                      # Risk-free rate (DAILY rate, already /365), shape: (batch_size, 1)
                      rf: torch.Tensor,
                      # Learned quantiles of return distribution, shape: (batch_size, num_quantiles)
                      varpi_q: torch.Tensor,
                      # Grid points for PDF interpolation, shape: (batch_size, grid_size)
                      varpi_grid: torch.Tensor,
                      # PDF values at grid points, shape: (batch_size, grid_size)
                      varpi_pdf: torch.Tensor,
                      # Quantile levels (e.g., [0.01, 0.02, ..., 0.99]), shape: (num_quantiles,)
                      var_rho_taus: torch.Tensor,
                      ) -> torch.Tensor:
        """
        Compute squared PDE residual for a single Monte Carlo evaluation.

        The PDE being enforced is:
        ∂V'/∂t' + (1/2)(S')²ρ²∂²V'/∂(S')² + rf·S'·∂V'/∂S' - rf·V' = 0

        Where ρ = varrho(t', varpi_q) is a stochastic coefficient computed from the
        learned return distribution. The randomness for ρ is generated internally
        in calc_varrho (z ~ N(0,1) is sampled inside that function).

        Returns:
            Mean squared residual over batch: (1/N) * Σ(residual²)
        """
        # Forward pass: Get model prediction V'(S', t', rf, varpi_q)
        # Shape: (batch_size, 1)
        V_prime = model(S_prime, t_prime, rf, varpi_q)

        # Compute stochastic volatility coefficient ρ = varrho(t', varpi_q)
        # This coefficient adapts based on the learned return distribution.
        # calc_varrho generates z ~ N(0,1) internally for Monte Carlo integration.
        # Shape: (batch_size, 1)
        varrho = calc_varrho(var_rho_taus, varpi_q, varpi_grid,
                             varpi_pdf, t_prime)

        # Compute first-order partial derivatives using autograd
        # V'_S = ∂V'/∂S', V'_tau = ∂V'/∂t'
        # Both shape: (batch_size, 1)
        # Ensure grad_outputs is on the same device as V_prime
        grad_outputs = torch.ones_like(V_prime, device=V_prime.device)
        grads = torch.autograd.grad(
            outputs=V_prime,
            inputs=(S_prime, t_prime),
            grad_outputs=grad_outputs,
            create_graph=True,  # Need for second derivatives
            retain_graph=True
        )
        V_prime_S = grads[0]      # ∂V'/∂S'
        V_prime_tau = grads[1]   # ∂V'/∂t'

        # Compute second-order partial derivative: V'_SS = ∂²V'/∂(S')²
        # Shape: (batch_size, 1)
        # Ensure grad_outputs is on the same device as V_prime_S
        grad_outputs_S = torch.ones_like(V_prime_S, device=V_prime_S.device)
        V_prime_S2 = torch.autograd.grad(
            outputs=V_prime_S,
            inputs=S_prime,
            grad_outputs=grad_outputs_S,
            create_graph=True,
            retain_graph=True
        )[0]

        # Build PDE residual: r = ∂V'/∂t' + (1/2)(S')²ρ²∂²V'/∂(S')² + rf·S'·∂V'/∂S' - rf·V'
        term1 = V_prime_tau                                    # ∂V'/∂t'
        term2 = 0.5 * (S_prime ** 2) * (varrho ** 2) * V_prime_S2  # (1/2)(S')²ρ²∂²V'/∂(S')²
        term3 = rf * S_prime * V_prime_S                      # rf·S'·∂V'/∂S'
        term4 = -rf * V_prime                                 # -rf·V'

        residual = term1 + term2 + term3 + term4

        # Return mean squared residual over batch
        residual_loss = torch.mean(residual ** 2)
        return residual_loss

    def payoff_loss(self, model, S_prime, t_max, rf, varpi_q):
        """
        Terminal boundary condition: At expiration (t'=0), option value = payoff.
        Must be implemented by subclass.
        """
        raise NotImplementedError("payoff_loss method must be implemented.")

    def under_loss(self, model, S_min, t_prime, rf, varpi_q):
        """
        Lower boundary condition: At S'=0, option value should be 0.
        Must be implemented by subclass.
        """
        raise NotImplementedError("under_loss method must be implemented.")

    def over_loss(self, model, S_max, t_prime, rf, varpi_q):
        """
        Upper boundary condition: At S'=S_max, option value should approach intrinsic value.
        Must be implemented by subclass.
        """
        raise NotImplementedError("over_loss method must be implemented.")

    def inequality_loss(self, model, S_prime, t_prime, rf, varpi_q):
        """
        American option constraint: V'(S', t') >= max(S' - 1, 0) everywhere.
        Penalizes violations of this inequality.
        Must be implemented by subclass.
        """
        raise NotImplementedError("inequality_loss method must be implemented.")

    def forward(self, model, sampled_data, taus, mc_samples):
        """
        Compute total training loss by combining PDE residual and boundary conditions.

        Process:
        1. For mc_samples iterations, compute squared PDE residual (averaged over batch)
        2. Each residual computation calls calc_varrho, which generates z ~ N(0,1) internally
        3. Average residual losses over all Monte Carlo iterations
        4. Compute boundary condition losses
        5. Combine with weights to get total loss

        Args:
            model: qStorm neural network model
            sampled_data: Dict containing:
                - S_prime: (batch_size, 1) normalized stock prices
                - t_prime: (batch_size, 1) time-to-expiration in days
                - rf: (batch_size, 1) daily risk-free rate
                - varpi_q: (batch_size, num_quantiles) learned quantiles
                - varpi_grids: (batch_size, grid_size) PDF grid points
                - varpi_pdfs: (batch_size, grid_size) PDF values
            taus: (num_quantiles,) quantile levels [0.01, 0.02, ..., 0.99]
            mc_samples: Number of Monte Carlo samples for stochastic term

        Returns:
            Dict with individual loss components and total loss
        """
        # Extract data from sampled_data dict
        S_prime = sampled_data["S_prime"]          # (batch_size, 1)
        t_prime = sampled_data["t_prime"]          # (batch_size, 1) - in DAYS
        rf = sampled_data["rf"]                    # (batch_size, 1) - DAILY rate
        varpi_q = sampled_data["varpi_q"]         # (batch_size, num_quantiles)
        varpi_grid = sampled_data["varpi_grids"]  # (batch_size, grid_size)
        varpi_pdf = sampled_data["varpi_pdfs"]     # (batch_size, grid_size)

        # Define boundary points - ensure on same device as inputs
        device = S_prime.device

        # Ensure taus is a tensor on the correct device
        if not isinstance(taus, torch.Tensor):
            taus = torch.tensor(taus, device=device, dtype=torch.float32)
        else:
            taus = taus.to(device)
        S_min = torch.zeros_like(S_prime, device=device)  # S' = 0
        S_max = torch.ones_like(S_prime, device=device) * 5  # S' = 5

        # Monte Carlo integration over stochastic volatility coefficient ρ
        # Each call to residual_loss -> calc_varrho generates independent z ~ N(0,1)
        # We average over mc_samples independent evaluations
        total_residual_loss = 0
        for _ in range(mc_samples):
            # Each iteration gets independent randomness from calc_varrho
            residual_loss = self.residual_loss(
                model=model,
                S_prime=S_prime,
                t_prime=t_prime,
                rf=rf,
                varpi_q=varpi_q,
                varpi_grid=varpi_grid,
                varpi_pdf=varpi_pdf,
                var_rho_taus=taus
            )
            total_residual_loss += residual_loss

        # Average over Monte Carlo samples
        # This computes: E_z[mean(residual²)] = (1/mc_samples) * Σ mean(residual²)
        # where z ~ N(0,1) is generated inside calc_varrho
        total_residual_loss = total_residual_loss / mc_samples

        # Terminal boundary condition: At expiration (t' = 0)
        # Ensure on same device as inputs
        t_max = torch.zeros_like(t_prime, device=t_prime.device)
        payoff_loss = self.payoff_loss(model, S_prime, t_max, rf, varpi_q)

        # Lower boundary condition: At S' = 0
        under_loss = self.under_loss(model, S_min, t_prime, rf, varpi_q)

        # Upper boundary condition: At S' = S_max
        over_loss = self.over_loss(model, S_max, t_prime, rf, varpi_q)

        # American option inequality constraint: V' >= max(S' - 1, 0)
        inequality_loss = self.inequality_loss(model, S_prime, t_prime, rf, varpi_q)

        # Weighted combination of all loss components
        total_loss = (self.lambda_p * payoff_loss +
                      self.lambda_u * under_loss +
                      self.lambda_o * over_loss +
                      self.lambda_i * inequality_loss +
                      total_residual_loss)

        # Unweighted loss (for monitoring/debugging)
        unadjusted_loss = (payoff_loss + under_loss + over_loss +
                           inequality_loss + total_residual_loss)

        losses = {
            "total_loss": total_loss,              # Weighted total loss (used for backprop)
            "unadjusted_loss": unadjusted_loss,    # Unweighted sum (for monitoring)
            "payoff_loss": payoff_loss,            # Terminal condition loss
            "under_loss": under_loss,              # Lower boundary loss
            "over_loss": over_loss,                # Upper boundary loss
            "inequality_loss": inequality_loss,    # American constraint violation
            "residual_loss": total_residual_loss   # Monte Carlo averaged PDE residual
        }
        return losses


class CallStormTrainer(StormTrainer):
    """
    Implementation of boundary conditions for American call options.

    For call options:
    - Terminal payoff: V'(S', 0) = max(S' - 1, 0)
    - Lower boundary: V'(0, t') = 0 (option worthless if stock = 0)
    - Upper boundary: V'(S_max, t') ≈ S_max - 1 (deep ITM approaches intrinsic)
    - Inequality: V'(S', t') >= max(S' - 1, 0) (American constraint)
    """

    def payoff_loss(self, model, S_prime, t_max, rf, varpi_q):
        """
        Terminal condition: At expiration (t'=0), call option value = max(S' - 1, 0).

        Computes: mean((V'(S', 0) - max(S' - 1, 0))²)
        """
        assert torch.all(t_max == 0), "t_max must be 0 for terminal condition"

        # Payoff function: max(S' - 1, 0) = ReLU(S' - 1)
        payoff = torch.relu(S_prime - 1.0)

        # Model prediction at expiration
        V_prime = model(S_prime, t_max, rf, varpi_q)

        # Mean squared error between model and payoff
        loss = torch.mean((V_prime - payoff)**2)
        return loss

    def under_loss(self, model, S_min, t_prime, rf, varpi_q):
        """
        Lower boundary condition: At S'=0, call option value must be 0.

        Computes: mean(V'(0, t')²)
        This enforces that if stock price is zero, option is worthless.
        """
        V_prime = model(S_min, t_prime, rf, varpi_q)
        return torch.mean((V_prime)**2)

    def over_loss(self, model, S_max, t_prime, rf, varpi_q):
        """
        Upper boundary condition: At S'=S_max, call option value ≈ S_max - 1.

        For deep ITM calls, the option value approaches intrinsic value.
        Note: Discounting term exp(-rf * t') was removed (as noted in comment).
        This assumes S_max is large enough that discounting is negligible.

        Computes: mean((V'(S_max, t') - (S_max - 1))²)
        """
        V_prime = model(S_max, t_prime, rf, varpi_q)
        # Target value: intrinsic value (no discounting applied)
        payout = S_max - 1.0
        loss = torch.mean((V_prime - payout)**2)
        return loss

    def inequality_loss(self, model, S_prime, t_prime, rf, varpi_q):
        """
        American call option constraint: V'(S', t') >= max(S' - 1, 0).

        For American options, the option value must always be >= immediate exercise value.
        This loss penalizes violations of this constraint.

        Computes: mean(max(max(S' - 1, 0) - V'(S', t'), 0)²)
        Only penalizes when exercise_value > V' (i.e., when constraint is violated).
        """
        # Model prediction
        V_prime = model(S_prime, t_prime, rf, varpi_q)

        # Immediate exercise value: max(S' - 1, 0) = ReLU(S' - 1)
        exercise_value = F.relu(S_prime - 1.0)

        # Constraint: V' >= exercise_value, so violation = exercise_value - V'
        # If violation > 0, constraint is violated
        violation = exercise_value - V_prime

        # Only penalize positive violations (clamp negative values to 0)
        # This gives: penalty = max(violation, 0) = ReLU(violation)
        penalty = F.relu(violation)

        # Return mean squared penalty
        return torch.mean(penalty ** 2)
