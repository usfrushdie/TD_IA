import random
import gym
import numpy as np
import matplotlib.pyplot as plt
import time

def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    Met à jour la table Q en suivant la formule de Q-learning.
    """
    # Formule de mise à jour de Q-learning
    Q[s, a] = Q[s, a] + alpha * (r + gamma * np.max(Q[sprime]) - Q[s, a])
    return Q

def epsilon_greedy(Q, s, epsilon):
    """
    Implémente la politique epsilon-greedy pour choisir une action.
    """
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Exploration : action aléatoire
    else:
        return np.argmax(Q[s])  # Exploitation : meilleure action connue

if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="human")
    env.reset()
    env.render()

    # Initialisation de la table Q
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    # Hyperparamètres
    alpha = 0.01  # Taux d'apprentissage
    gamma = 0.8   # Facteur de discount
    epsilon = 0.2 # Taux d'exploration

    # Configuration de l'entraînement
    n_epochs = 20  # Nombre d'épisodes
    max_itr_per_epoch = 100  # Nombre maximum d'itérations par épisode
    rewards = []

    for e in range(n_epochs):
        r = 0
        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            # Choisir une action avec epsilon-greedy
            A = epsilon_greedy(Q=Q, s=S, epsilon=epsilon)

            # Effectuer l'action et observer le résultat
            Sprime, R, done, _, info = env.step(A)

            # Cumul des récompenses
            r += R

            # Mise à jour de la table Q
            Q = update_q_table(Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma)

            # Mettre à jour l'état
            S = Sprime

            if done:  # Si l'épisode est terminé
                break

        # Afficher la récompense par épisode
        print("Episode #", e, " : Reward = ", r)

        # Stocker la récompense pour chaque épisode
        rewards.append(r)

    # Moyenne des récompenses
    print("Average reward = ", np.mean(rewards))

    # Tracé des récompenses par épisode
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.show()

    print("Training finished.\n")

    env.close()
