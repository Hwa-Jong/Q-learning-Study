import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm

def get_model(num_actions):
    model = nn.Sequential(*[
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, num_actions),
    ])
    return model


def main():
    # 그리드 환경 정의
    grid_size = 5
    num_states = grid_size*grid_size
    num_actions = 4  # 상, 하, 좌, 우
    goal_state = (4,4)

    # 하이퍼파라미터
    learning_rate = 0.8
    discount_factor = 0.95
    num_episodes = 1000


    model = get_model(num_actions)  

    criterion = nn.L1Loss()
    optim = torch.optim.Adam(lr=1e-3, params=model.parameters())

    
    # 목표 지점까지의 최단 경로 학습
    for episode in tqdm(range(num_episodes)):
        state = (0,0)
        state_arr = torch.from_numpy( np.zeros((1,1,grid_size,grid_size), dtype=np.float32))
        state_arr[:,:,state[0], state[1]] = 1
        done = False

        while state != goal_state:
            with torch.no_grad():
                q_values = model(state_arr)

            # ε-탐욕적 정책 (탐험과 이용을 결합)
            if np.random.rand() < 0.5:
                action = np.random.choice(num_actions)  # 무작위 행동 선택
            else:
                action = torch.argmax(q_values[0].detach()).item()  # Q 값이 가장 높은 행동 선택

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

            next_state_arr = torch.from_numpy( np.zeros((1,1,grid_size,grid_size), dtype=np.float32))
            next_state_arr[:,:,next_state[0], next_state[1]] = 1


            # 보상 및 Q 값 업데이트
            reward = -1 if next_state != goal_state else 0  # 목표 지점에 도달하면 보상 0, 그 외에는 -1

            # Q 값 업데이트 (CNN 학습)
            target = reward + discount_factor * torch.max(model(next_state_arr).detach()).item()            
            q_values[0,action] = target

            optim.zero_grad()
            output = model(state_arr)
            loss = criterion(q_values, output)
            loss.backward()
            optim.step()

            # 상태 업데이트
            state = next_state
            state_arr = next_state_arr


    # 학습된 Q 함수를 사용하여 최단 경로 찾기
    state = (0, 0)
    path = [state]
    while state != goal_state:
        state_arr = torch.from_numpy( np.zeros((1,1,grid_size,grid_size), dtype=np.float32))
        state_arr[:,:,state[0], state[1]] = 1
        
        with torch.no_grad():
            q_values = model(state_arr)
        action = torch.argmax(q_values[0].detach()).item()

        if action == 0:  # 상
            state = (max(state[0] - 1, 0), state[1])
        elif action == 1:  # 하
            state = (min(state[0] + 1, grid_size - 1), state[1])
        elif action == 2:  # 좌
            state = (state[0], max(state[1] - 1, 0))
        elif action == 3:  # 우
            state = (state[0], min(state[1] + 1, grid_size - 1))
        path.append(state)
        if len(path) > 100:
            break

    # 결과 출력
    print('%d 에피소드 결과'%num_episodes)
    if len(path) > 100:
        print("인공지능 결과 경로 탐색 실패")
    else:
        print("인공지능 결과 최단 경로:", path)


if __name__ =='__main__':
    main()