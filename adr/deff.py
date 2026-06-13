import torch
from torch.func import functional_call
from .utils import get_target_params, compute_jacobian, spectral_scale


def get_d_eff(
    model,
    pde_fn,
    target_layers,
    x_pde,
    x_bc=None,
    mode="all",
    rcond=1e-10,
    engine="auto",
    use_base_state=True,
    lam=1.0,
):
    """Compute effective degrees of freedom (d_eff) for the specified layers.

    d_eff measures how many unconstrained directions remain in the target layers
    given the PDE and/or BC constraints. Use it to identify which layers have
    capacity to absorb new boundary data without disrupting the learned physics.

    Parameters
    ----------
    model          : nn.Module
    pde_fn         : callable(model, x) -> residual [N, 1] or [N]
                     must use create_graph=True for spatial derivatives internally
    target_layers  : list of int or str
    x_pde          : tensor [N, d_in] — collocation points for PDE and spatial eval
    x_bc           : tensor [K, d_in] — BC collocation points (required unless mode='pde')
    mode           : 'pde', 'bc', 'total', or 'all'
    rcond          : singular value / eigenvalue cutoff for rank truncation
    engine         : 'auto' (default) — ntk when d > N+M, parameter otherwise
                     'ntk'           — dual formulation O((N+M)^3)
                     'parameter'     — primal formulation O(d^3)
    use_base_state : if True (default) and model has _adr_base_state, evaluate at the
                     stored base state; if False, evaluate at the current parameter state
    lam            : float (default 1.0) — scales the constraint Jacobian J; in the NTK
                     engine K = [Psi_n; lam*J], in the parameter engine A = Q + lam^2 * J^T J

    Returns
    -------
    dict {'pde': float, 'bc': float, 'total': float}  if mode == 'all'
    float                                               otherwise
    """
    was_training = model.training
    model.eval()
    current_state = {n: p.data.clone() for n, p in model.named_parameters()}
    if use_base_state and hasattr(model, "_adr_base_state"):
        # want to ensure that the base state is used for the Jacobian computation
        # not the current state which may have been updated by some adaptation
        for name, p in model.named_parameters():
            if name in model._adr_base_state:
                p.data.copy_(model._adr_base_state[name])
    try:
        return _compute(model, pde_fn, target_layers, x_pde, x_bc, mode, rcond, engine, lam)
    finally:
        for name, p in model.named_parameters():
            p.data.copy_(current_state[name])
        model.train(was_training)


def _compute(model, pde_fn, target_layers, x_pde, x_bc, mode, rcond, engine, lam=1.0):
    params = get_target_params(model, target_layers)
    d_params = sum(p.numel() for p in params)

    # Partition parameters: target ones are passed to jacrev, static ones are captured
    all_named = dict(model.named_parameters())
    param_ids = {id(p) for p in params}
    target_names = [n for n, p in all_named.items() if id(p) in param_ids]
    static_dict = {n: p for n, p in all_named.items() if id(p) not in param_ids}

    def _make_pd(*p):
        return {**static_dict, **dict(zip(target_names, p))}

    x_pde_t = x_pde.clone().requires_grad_(True)
    N = len(x_pde_t)

    def _fwd_model_pde(*p):
        return functional_call(model, _make_pd(*p), x_pde_t).view(-1)

    Psi_n = compute_jacobian(_fwd_model_pde, params) / N**0.5

    J_pde_n = None
    if mode in ("pde", "total", "all"):

        def _fwd_pde(*p):
            return pde_fn(model, x_pde_t, param_dict=_make_pd(*p)).view(-1)

        J_pde_raw = compute_jacobian(_fwd_pde, params)
        J_pde_n = J_pde_raw / len(J_pde_raw) ** 0.5

    J_bc_n = None
    if mode in ("bc", "total", "all"):
        x_bc_t = x_bc.clone().requires_grad_(True)

        def _fwd_bc(*p):
            return functional_call(model, _make_pd(*p), x_bc_t).view(-1)

        J_bc_raw = compute_jacobian(_fwd_bc, params)
        J_bc_n = J_bc_raw / len(J_bc_raw) ** 0.5

    def _deff_ntk(J):
        K = torch.cat([Psi_n, lam * J], dim=0)
        G = K @ K.T
        scale = G.diagonal().mean().clamp(min=1e-30).sqrt()
        G_pinv = torch.linalg.pinv(G / scale, rcond=rcond, hermitian=True) / scale
        Pi_G = G @ G_pinv
        P_11 = Pi_G[:N, :N]
        return torch.sum(P_11**2).item()

    def _deff_parameter(J):
        Q = Psi_n.T @ Psi_n
        A = Q + lam**2 * J.T @ J
        scale = A.diagonal().mean().clamp(min=1e-30).sqrt()
        A_pinv = torch.linalg.pinv(A / scale, rcond=rcond, hermitian=True) / scale
        QA = Q @ A_pinv
        return (QA * QA.T).sum().item()

    def _route(dual_dim):
        if engine == "ntk":
            return _deff_ntk
        if engine == "parameter":
            return _deff_parameter
        return _deff_parameter if d_params <= dual_dim else _deff_ntk

    results = {}

    if mode in ("pde", "all"):
        results["pde"] = _route(N + len(J_pde_n))(J_pde_n)

    if mode in ("bc", "all"):
        results["bc"] = _route(N + len(J_bc_n))(J_bc_n)

    if mode in ("total", "all"):
        scale_pb = spectral_scale(J_pde_n, J_bc_n)
        J_total = torch.cat([J_pde_n, scale_pb**0.5 * J_bc_n], dim=0)
        results["total"] = _route(N + len(J_pde_n) + len(J_bc_n))(J_total)

    return results if mode == "all" else results[mode]
