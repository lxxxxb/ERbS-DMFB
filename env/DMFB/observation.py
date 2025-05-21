import numpy as np
# state data prepare

def fov_origon(x,y,w,l,fov):
    hf=fov//2
    ox=x-hf
    if x-hf<0:
        ox = 0
    elif ox+fov>w:
        ox = w-fov
    oy=y-hf
    if y-hf<0:
        oy = 0
    elif oy+fov>l:
        oy = l-fov
    return ox,oy

def line_intersects_rect(line_start, line_end, rect_top_left, rect_bottom_right):
    rect_lines = [
        (rect_top_left, (rect_top_left[0], rect_bottom_right[1])),
        ((rect_top_left[0], rect_bottom_right[1]), rect_bottom_right),
        (rect_bottom_right, (rect_bottom_right[0], rect_top_left[1])),
        ((rect_bottom_right[0], rect_top_left[1]), rect_top_left),
    ]

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def do_intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    for rect_line in rect_lines:
        if do_intersect(line_start, line_end, rect_line[0], rect_line[1]):
            return True
    return False

def manhatten_dist(A, B):
    return abs(A[0] - B[0]) + abs(A[1] - B[1])

def find_block(source, goal, blocks):
    # def calculate_angle(vector1, vector2):
    #     dot_product = np.dot(vector1, vector2)
    #     norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    #     cos_theta = np.clip(dot_product / norm_product, -1.0, 1.0)
    #     return np.arccos(cos_theta)



    # intersections = np.zeros((2,2))
    block4d = None
    distmin = float('Inf')
    for block in blocks:
        rect_top_left = (block.x_min, block.y_min)
        rect_bottom_right = (block.x_max, block.y_max)


        if line_intersects_rect(source, goal, rect_top_left, rect_bottom_right):
            middle=(rect_top_left[0]+rect_bottom_right[0])/2, (rect_top_left[1]+rect_bottom_right[1])/2
            dist=manhatten_dist(middle, source)

            if dist< distmin:
                distmin = dist
                block4d = [rect_top_left, rect_bottom_right]
                # 两条对角线的方向
                # diag1 = np.array([
                #     (rect_top_left[0] - source[0], rect_top_left[1] - source[1]),
                #     (rect_bottom_right[0] - source[0], rect_bottom_right[1] - source[1])
                # ])
                # diag2 = np.array([
                #     (rect_top_right[0] - source[0], rect_top_right[1] - source[1]),
                #     (rect_bottom_left[0] - source[0], rect_bottom_left[1] - source[1])
                # ])

    # 计算夹角
    # line_vector = (goal[0] - source[0], goal[1] - source[1])
    # angle1 = max(calculate_angle(line_vector, diag1[0]), calculate_angle(line_vector, diag1[1]))
    # angle2 = max(calculate_angle(line_vector, diag2[0]), calculate_angle(line_vector, diag2[1]))

    # if angle1 > angle2:
    #     intersections=diag1
    # else:
    #     intersections=diag2
    # if not result:
    #     raise ValueError
    return block4d

def guide_line(droplet, blocks):
    source = droplet.pos
    goal = droplet.des
    def first3points(source, rect_top_left, rect_bottom_right):
        rect_top_right = (rect_bottom_right[0], rect_top_left[1])
        rect_bottom_left = (rect_top_left[0], rect_bottom_right[1])
        result = [rect_top_left, rect_bottom_right, rect_top_right, rect_bottom_left]
        result = [(x - source[0], y - source[1]) for (x,y) in result]
        result = sorted(result, key=lambda x: abs(x[0]) + abs(x[1]), reverse=False)
        result = result[0:3]
        return result
    if droplet.block is not None:
        rect_top_left, rect_bottom_right= droplet.block
        if line_intersects_rect(source, goal, rect_top_left, rect_bottom_right):
            return first3points(source, rect_top_left, rect_bottom_right)
    droplet.block = find_block(source, goal, blocks)
    if droplet.block is not None:
        rect_top_left, rect_bottom_right = droplet.block
        return first3points(source, rect_top_left, rect_bottom_right)
    return [(0,0),(0,0),(0,0)]

def observation(s_c, state_d, last_action, fovset):
    '''
     s_c: chip status;
            width * length;
     state_d: [for type 0: (0, x, y, dx, dy, partner),
              for type 1: (1, x, y, mix_percent, headto, 0)]
     format of gloabal observed
     2-w-l
     Droplets               in layer 0  [id  0]
     Obstacles              in layer 1  [x   1]
     '''
    width,length=s_c.shape
    fovs_0=[]
    linears_0=[] #传递给type0的attention网络的
    fovs_1=[]
    linears_1=[]
    attmatrix=[]
    # add droplets on chip
    gobs=np.zeros((width,length))
    for i in range(state_d.shape[0]):
        gobs[state_d[i, 1], state_d[i, 2]] = i+1

    for agent_i in range(state_d.shape[0]):
        s_d=state_d[agent_i]
        type = s_d[0]


        fov = fovset[type]  # 正方形fov边长
        obs_i = np.zeros((2, fov, fov))
        ox, oy = fov_origon(s_d[1], s_d[2], width, length, fov)
        obs_i[0] = s_c[ox:ox + fov, oy:oy + fov]  # block 取fov所在区域
        obs_i[1] = gobs[ox:ox + fov, oy:oy + fov]  #
        attmatrix.append(np.unique[obs_i[1]])
        obs_i[1][obs_i[1] == agent_i] = -1 # 标记一下当前液滴的位置


        if type == 0:
            if s_d[5] != 0:
                obs_i[1][obs_i[1] == s_d[5]] = 0 # 删除partner液滴
            fovs_0.append(obs_i)
            linears_0.append(s_d[1:5]+last_action[agent_i]) #不一定对
            continue
        elif type == 1:
            headto=s_d[4]
            if headto == 2:
                obs_i = np.flip(obs_i, 1)
            elif headto == 3:
                obs_i = np.rot90(obs_i, axes=(1, 2))
            elif headto == 4:
                obs_i = np.rot90(obs_i, axes=(2, 1))
            fovs_1.append(obs_i)
            linears_1.append(s_d[1:5]+last_action[agent_i])
            continue
        elif type == 2:
            linears_0.append(s_d[1:5]+last_action[agent_i]) # store液滴和type0液滴暂时共用同一个attention网络好了，相当于到了终点的type1? 毕竟只有测试时才会用到
            continue
        else:
            raise ValueError

    return fovs_0, fovs_1, linears_0, linears_1, attmatrix

import torch
# 针对一个transition的所有episode提取input
def get_q_values(s_c, s_d, u_onehot, max_episode_len, n_agents, fovset):
    for transition_idx in max_episode_len:
        s_ci = s_c[:,transition_idx]
        s_di = s_d[:,transition_idx]
        if transition_idx == 0:
            u_onehoti = torch.zeros_like(u_onehot[:,transition_idx])
        else:
            u_onehoti = u_onehot[:,transition_idx-1]
        records=[[]]*5
        for ep in s_ci.shape[0]: #以后看看能不能批处理（不用for了）
            record = observation(s_ci[ep], s_di[ep], u_onehoti[ep], fovset)
            for i in range(5):
                records[i].append(record[i])
        for i in range(4): # matrix不用
            records[i] = torch.cat(records[i], dim=2).flatten(end_dim=1) # 这里也之后再试吧，肯定不对
        fovs_0, fovs_1, linears_0, linears_1, attmatrix = records
        # contex0=attnet(fovs_0)
        # contex1=attnet(fovs_1)
        # 还得把contex0和contex1根据s_di交叉在一起拼起来愁人。
    return fovs_0, fovs_1, linears_0, linears_1, attmatrix

def prepare_data(episodes, fov):
    # 储存前计算什么
    # 现有最简单的带att的方案就是计算距离矩阵然后把相关的投进全连接--》
    # 计算一个距离矩阵然后删除大于fov的，再反向归一化
    size = len(episodes)
    eplen, n_agents = episodes[0]['u'].shape[1:3] #每一次的eplen可能不一样
    points = torch.zeros((size,eplen+1, n_agents,2))
    for i in range(size):
        points[i] = torch.index_select(episodes[i]['s_d'],-1,torch.tensor([1,2]))
    dis = EDM_torch(points)
    dis[dis > fov] = fov+1
    scaled_d =1 - (dis/fov+1) #相关性矩阵
    scaled_d = scaled_d / torch.sum(scaled_d, dim=-1, keepdim=True) #对每一行归一化
    for i in range(size):
        episodes[i]['s_d'] = torch.cat([episodes[i]['s_d'], scaled_d[i]])
    return episodes

def state2obs(episode, fov):
    # 储存前计算什么
    # 这是对每一条episode计算的版本，就不用一个step一个step都算了
    # 现有最简单的带att的方案就是计算距离矩阵然后把相关的投进全连接--》
    # 计算一个距离矩阵然后删除大于fov的，再归一化
    points = torch.index_select(episode['s_d'],-1,torch.tensor([1,2]))
    partner = torch.index_select(episode['s_d'],-1,torch.tensor([5])).squeeze(dim=-1)-1  #(1, eplen, n_agents)每个液滴对应partner的索引
    dis = EDM_torch(points)
    dis[dis > fov] = fov+1
    scaled_d =1 - (dis/fov+1) #相关性矩阵
    scaled_d = scaled_d / torch.sum(scaled_d, dim=-1, keepdim=True) #对每一行归一化

    # 获取partner中不等于-1的元素的索引
    indices = torch.nonzero(partner != -1, as_tuple=True)
    # 让scaled_d中对应partner的元素为0，这样液滴就相当于看不见patner了，只是知道自己的目标
    scaled_d[indices[0], indices[1], indices[2], partner[indices]] = 0

    #加last action
    shape = episode['u_onehot'].shape
    # 在开头添加一列全0的数
    last_action = torch.zeros((shape[0], shape[1] + 1, *shape[2:]))
    last_action[:, 1:, :, :] = episode['u_onehot']
    episode['s_d'] = torch.cat([episode['s_d'], last_action], dim=-1) # 拼上last_action

    episode['s_d'] = torch.cat([episode['s_d'], scaled_d], dim=-1) # 把自己和其他液滴的相关性矩阵拼起来
    return episode

def state2obs_one_step_torch(state,last_action, fov):
    # 这是对每一step计算的版本
    # 现有最简单的带att的方案就是计算距离矩阵然后把相关的投进全连接--》
    # 计算一个距离矩阵然后删除大于fov的，再归一化
    points = torch.index_select(state,-1,torch.tensor([1,2]))
    partner = torch.index_select(state,-1,torch.tensor([5])).squeeze(dim=-1)-1  #(1, eplen, n_agents)每个液滴对应partner的索引
    dis = EDM_torch(points)
    dis[dis > fov] = fov+1
    scaled_d =1 - (dis/fov+1) #相关性矩阵
    scaled_d = scaled_d / torch.sum(scaled_d, dim=-1, keepdim=True) #对每一行归一化

    # 获取partner中不等于-1的元素的索引
    indices = torch.nonzero(partner != -1, as_tuple=True)
    # 让scaled_d中对应partner的元素为0，这样液滴就相当于看不见patner了，只是知道自己的目标
    scaled_d[indices[0], indices[1], partner[indices]] = 0

    #加last action
    # shape = experience['u_onehot'].shape
    # # 在开头添加一列全0的数
    # last_action = torch.zeros((shape[0], shape[1] + 1, *shape[2:]))
    # last_action[:, 1:, :, :] = experience['u_onehot']
    state = torch.cat([state, last_action], dim=-1) # 拼上last_action

    state = torch.cat([state, scaled_d], dim=-1) # 把自己和其他液滴的相关性矩阵拼起来
    return state


def EDM_torch(c):
    # c:shape--(*, points,2)
    c1 = torch.unsqueeze(c, dim=-2) # (points,1,2)
    c2 = torch.unsqueeze(c, dim=-3)   # (1,points,2)
    # 如果是manhadon距离是torch.sum(abs(c1-c2), dim=-1)
    # 这里用的是欧式距离
    return torch.sum((c1-c2)**2, dim=-1)**0.5  #c1-c2--(points,points,2); dim=-1 对2求和

if __name__ == "__main__":
    from classes import Block, Droplet_T1
    source1 = (1, 1)
    goal1 = (8, 8)
    blocks1 = [
        Block(3, 5, 0, 2),  # 位于路径之外
        Block(6, 9, 4, 6)  # 位于路径之外
    ]
    source2 = (0, 0)
    goal2 = (10, 10)
    blocks2 = [
        Block(3, 6, 3, 6),  # 位于路径上的障碍
        Block(8, 9, 1, 4)  # 位于路径之外
    ]
    source3 = (0, 0)
    goal3 = (15, 10)
    blocks3 = [
        Block(4, 6, 4, 7),  # 位于路径上的障碍
        Block(10, 12, 3, 6),  # 位于路径上的障碍
        Block(14, 16, 8, 10),  # 不在路径上的障碍
    ]
    test_cases = [
        {"source": (1, 1), "goal": (8, 8), "blocks": blocks1, "description": "No obstacle on the line"},
        {"source": (0, 0), "goal": (10, 10), "blocks": blocks2, "description": "One obstacle on the line"},
        {"source": (0, 0), "goal": (15, 10), "blocks": blocks3, "description": "Multiple obstacles on the line"}
    ]

    for i, test in enumerate(test_cases):
        print(f"Test Case {i + 1}: {test['description']}")
        directions = guide_line(Droplet_T1(test["source"], test["goal"]), test["blocks"])
        print(directions)