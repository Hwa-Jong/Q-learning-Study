import numpy as np

def main():
    # 그리드 환경 정의
    grid_size = 5
    num_actions = 4  # 상, 하, 좌, 우
    goal_state = (4, 4)

    # Q 함수 초기화
    Q = np.zeros((grid_size, grid_size, num_actions))

    # 하이퍼파라미터
    learning_rate = 0.8
    discount_factor = 0.95
    num_episodes = 5000

    # 목표 지점까지의 최단 경로 학습
    for episode in range(num_episodes):
        state = (0, 0)  # 에이전트 초기 위치
        done = False

        while state != goal_state:
            # ε-탐욕적 정책 (탐험과 이용을 결합)
            if np.random.rand() < 0.5:
                action = np.random.choice(num_actions)  # 무작위 행동 선택
            else:
                action = np.argmax(Q[state[0], state[1], :])  # Q 값이 가장 높은 행동 선택

            # 다음 상태 계산
            next_state = state
            if action == 0:  # 상
                next_state = (max(state[0] - 1, 0), state[1])
            elif action == 1:  # 하
                next_state = (min(state[0] + 1, grid_size - 1), state[1])
            elif action == 2:  # 좌
                next_state = (state[0], max(state[1] - 1, 0))
            elif action == 3:  # 우
                next_state = (state[0], min(state[1] + 1, grid_size - 1))

            # 보상 및 Q 값 업데이트
            reward = -1 if next_state != goal_state else 0  # 목표 지점에 도달하면 보상 0, 그 외에는 -1
            Q[state[0], state[1], action] = (1 - learning_rate) * Q[state[0], state[1], action] + learning_rate * (reward + discount_factor * np.max(Q[next_state[0], next_state[1], :]))

            # 상태 업데이트
            state = next_state

    # 학습된 Q 함수를 사용하여 최단 경로 찾기
    state = (0, 0)
    path = [state]
    while state != goal_state:
        action = np.argmax(Q[state[0], state[1], :])
        if action == 0:
            state = (max(state[0] - 1, 0), state[1])
        elif action == 1:
            state = (min(state[0] + 1, grid_size - 1), state[1])
        elif action == 2:
            state = (state[0], max(state[1] - 1, 0))
        elif action == 3:
            state = (state[0], min(state[1] + 1, grid_size - 1))
        path.append(state)

    # 결과 출력
    print("최단 경로:", path)


if __name__ =='__main__':
    main()