from interface.WTA import WTA


class HybridWtaModule(WTA):
    """Ask when the step is a multiple of interval OR nav confidence < threshold."""
    def __init__(self, interval=4, threshold=0.6):
        self.interval = interval
        self.threshold = threshold

    def wta(self, t, prob, nav_outs):
        probs, _ = prob.max(1)
        probs_cpu = probs.cpu()
        return [(t % self.interval == 0) or (p < self.threshold) for p in probs_cpu]


class CappedConfidenceWtaModule(WTA):
    """Ask when nav confidence < threshold, but at most `cap` questions per
    episode. The cap bounds DTC, which matters because the score's E(D) term
    penalizes asking more questions than the GT dialog length (~1-2)."""

    def __init__(self, threshold=0.6, cap=2, min_step=0):
        self.threshold = threshold
        self.cap = cap
        self.min_step = min_step
        self.ask_counts = []
        self.blocked = None

    def set_blocked(self, blocked):
        """Entries whose question the caller will drop anyway.

        Without this the cap is charged for questions that never reach the
        guide, so an episode can exhaust its quota during a phase where dialog
        is suppressed and then be unable to ask when it matters.
        """
        self.blocked = list(blocked) if blocked is not None else None

    def wta(self, t, prob, nav_outs):
        probs, _ = prob.max(1)
        probs_cpu = probs.cpu()
        n = len(probs_cpu)
        if t == 0:
            self.ask_counts = [0] * n
        blocked = self.blocked if self.blocked and len(self.blocked) == n else None
        ask = []
        for i, p in enumerate(probs_cpu):
            if blocked is not None and blocked[i]:
                ask.append(False)
            elif t >= self.min_step and p < self.threshold and self.ask_counts[i] < self.cap:
                self.ask_counts[i] += 1
                ask.append(True)
            else:
                ask.append(False)
        return ask
