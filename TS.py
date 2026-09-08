"""
TS.py — Thompson Sampling for a Gaussian-bandit ("restaurant satisfaction") toy problem.

Patched bugs (compared to the version currently in the repo):
  1. `get_mu_from_current_distribution` had lost its `return samp_mu` statement
     (a stray edit turned it into a no-op that returned None), which crashed
     `max(post_samps)` in the main loop since None can't be compared.
  2. `draw_distributions` called `sns.kdeplot(...)` but seaborn was never
     imported (the import was commented out), so every call raised
     NameError. Replaced with a matplotlib-only kernel density estimate
     (via scipy.stats.gaussian_kde) so there's no seaborn dependency;
     falls back to a plain histogram if scipy isn't installed either.
"""
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import gaussian_kde
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


class restaurant:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.n = 0
        self.sum_satisfaction = 0

    def get_satisfaction_from_true_distribution(self):
        s = np.random.normal(self.mu, self.sigma)
        self.n += 1
        self.sum_satisfaction += s
        return s


class RestaurantThompsonSampler(restaurant):
    def __init__(self, mu, sigma):
        self.prior_mu_of_mu = 0
        self.prior_sigma_of_mu = 10000

        self.post_mu_of_mu = self.prior_mu_of_mu
        self.post_sigma_of_mu = self.prior_sigma_of_mu

        self.n = 0
        self.sum_satisfaction = 0
        super().__init__(mu, sigma)

    def get_mu_from_current_distribution(self):
        samp_mu = np.random.normal(self.post_mu_of_mu, self.post_sigma_of_mu)
        return samp_mu  # FIX: this return was missing in the repo version

    def update_current_distribution(self):
        self.post_sigma_of_mu = np.sqrt(
            (1 / self.prior_sigma_of_mu**2 + self.n / self.sigma**2)**-1)
        self.post_mu_of_mu = (self.post_sigma_of_mu**2) * (
            (self.prior_mu_of_mu / self.prior_sigma_of_mu**2)
            + (self.sum_satisfaction / self.sigma**2)
        )


def draw_distributions(R, i):
    plt.figure()
    xs = np.linspace(-10, 10, 400)
    for r in R:
        samps = np.random.normal(r.post_mu_of_mu, r.post_sigma_of_mu, 10000)
        if _HAVE_SCIPY and np.std(samps) > 1e-9:
            density = gaussian_kde(samps)(xs)
            plt.plot(xs, density)
            plt.fill_between(xs, density, alpha=0.3)
        else:
            plt.hist(samps, bins=60, density=True, alpha=0.4,
                     range=(-10, 10))
    plt.title('Iteration %s' % (i + 1), fontsize=20)
    plt.legend(['mu=%s' % (r.mu) for r in R], fontsize=10)
    plt.xlim(-10, 10)
    plt.xlabel('Average Satisfaction', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'ts_iteration_{i+1}.png', dpi=120)
    plt.close()


if __name__ == "__main__":
    num_restaurants = 10
    spacing = 0.33
    R = [RestaurantThompsonSampler(i * spacing, 1)
         for i in range(1, num_restaurants + 1)]

    for i in range(2000):
        if num_restaurants <= 10 and (i < 10 or (i < 100 and (i + 1) % 10 == 0) or ((i + 1) % 100 == 0)):
            draw_distributions(R, i)

        # get a sample from each posterior
        post_samps = [r.get_mu_from_current_distribution() for r in R]

        # index of distribution with highest satisfaction
        chosen_idx = post_samps.index(max(post_samps))

        # get a new sample from that distribution
        s = R[chosen_idx].get_satisfaction_from_true_distribution()

        # update that distribution's posterior
        R[chosen_idx].update_current_distribution()

    plt.figure(figsize=(5, 5))
    true_means = [r.mu for r in R]
    posterior_means = [r.post_mu_of_mu for r in R]
    plt.scatter(true_means, posterior_means)
    plt.plot(true_means, true_means, color='k', alpha=0.5, linestyle='--')
    plt.xlabel('True Mean', fontsize=16)
    plt.ylabel('Posterior Mean', fontsize=16)
    plt.tight_layout()
    plt.savefig('ts_true_vs_posterior.png', dpi=120)
    plt.close()
    print("Done. Saved ts_iteration_*.png and ts_true_vs_posterior.png")
