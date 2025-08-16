import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

# @title Basic Rule Extraction
def get_data(env, agent, num_episodes):
    X = np.zeros((500 * num_episodes,8))
    y = np.zeros((500 * num_episodes,1))

    scores = []
    i=0
    for game in range(num_episodes):

        state = env.reset()[0]
        done = False
        game_score = 0
        while not done:
            if i == len(X):
                break
            action = agent.act(state)
            X[i , :] = np.array(state)
            y[i] = action
            i += 1
            state, rew , truncated , terminated, _ = env.step(action)
            done = truncated or terminated
            game_score += rew
            if done:
                scores.append(game_score)
                break

    print(f"Average game score: {np.mean(scores)}")
    non_zero_X = ~np.all(X == 0, axis=1)
    filtered_X = X[non_zero_X]
    filtered_y = y[non_zero_X]

    return filtered_X, filtered_y

def get_rules(env , model, num_episodes, max_depth = 10):
    X_train, y_train = get_data(env, model, num_episodes = num_episodes)
    tree_clf = DecisionTreeClassifier(max_depth = max_depth)
    tree_clf.fit(X_train, y_train)
    return tree_clf

def print_rules(tree_classifier):
    tree_rules = export_text(tree_classifier, feature_names=['x_position', 'y_position', 'x_velocity', 'y_velocity' , 'angle' , 'angular velocity','leg1', 'leg2'])
    print("Decision Tree Rules:")
    print(tree_rules)

def test_classifier(env , tree_classifier , num_of_episodes):
    scores = []
    for game in range(num_of_episodes):
        state = env.reset()[0]
        done = False
        score = 0
        while not done:
            state = np.reshape(state,(1,env.observation_space.sample().shape[0]))
            action = tree_classifier.predict(state)
            next_state , reward , truncated , terminated , _ = env.step(int(action.item()))
            done = truncated or terminated
            state = next_state
            score += reward
            if done:
                break
        scores.append(score)

    mean_score = np.mean(scores)
    return mean_score