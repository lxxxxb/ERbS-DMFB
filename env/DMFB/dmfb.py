from .classes import *
import gym
import numpy as np
from pettingzoo.utils.env import ParallelEnv
import cv2

from copy import deepcopy
'''
DMFBs MARL enviroment created by Jessie
'''




class RoutingTaskManager:
    def __init__(self, chip, fov, oc=2, stall=False):
        self.width = chip.width
        self.length = chip.length
        self.chip=chip
        self.droplets = Droplets([])
        self.oc=oc
        if fov[0] > min(self.width, self.length):
            raise RuntimeError('Fov is too large')
        self.fov = fov
        self.stall = stall
        self.step_count = 0
        random.seed(datetime.now())


    def refresh(self):
        self.droplets.clear()
        self.chip.updateHealth()

    def restartforall(self):
        self.step_count = 0
        for d in self.droplets:
            d.refresh()

    def _Generate_Locations(self,n_droplets):
        Locations = randomXY(self.width, self.length, n_droplets)
        if n_droplets == 1:
            return Locations
        dis = compute_norm_squared_EDM(Locations) #计算生成的点之间的距离
        strided = np.lib.stride_tricks.as_strided
        s0, s1 = dis.strides
        m = dis.shape[0]
        out = strided(dis.ravel()[1:], shape=(m-1, m),
                      strides=(s0+s1, s1)).reshape(m, -1)
        while out.min() <= 2 or self.chip._isinsideBlocks(Locations):
            Locations = randomXY(self.width, self.length, n_droplets)
            dis = compute_norm_squared_EDM(Locations)
            out = strided(dis.ravel()[1:], shape=(
                m-1, m), strides=(s0+s1, s1)).reshape(m, -1)
        return Locations

    def moveDroplets(self, actions):
        occupied=self.last_frame.copy()
        if len(actions) != self.droplets.__len__():
            raise RuntimeError("The number of actions is not the same"
                               " as n_droplets")

        self.step_count += 1
        rewards = []
        dones = self.getTaskStatus()
        constraints=0
        for i in range(self.droplets.__len__()):
            reward, constraint = self.moveOneDroplet(i, actions[i], occupied)
            rewards.append(reward)
            constraints += constraint
        rewards = np.array(rewards)-2 * constraints
        terminated = self.check_finish(self.step_count)
        return rewards, constraints, terminated

    def typeindex(self):
        index_type0=[]
        index_type1=[]
        for i, d in enumerate(self.droplets):
            if d.type==0:
                index_type0.append(i)
            elif d.type ==1:
                index_type1.append(i)
        return index_type0, index_type1

    def _isinvalidaction(self):
        position = np.zeros((self.droplets.__len__(), 2))
        for i, d in enumerate(self.droplets):
            position[i][0], position[i][1] = d.x, d.y
        dis = compute_norm_squared_EDM(position)
        m = dis.shape[0]
        strided = np.lib.stride_tricks.as_strided
        s0, s1 = dis.strides
        out = strided(dis.ravel()[1:], shape=(m-1, m),
                      strides=(s0+s1, s1)).reshape(m, -1)
        if out.min() == 0:
            return True
        else:
            return False

    def moveOneDroplet(self, droplet_index, action, occupied):
        constraint = 0
        if droplet_index >= self.droplets.__len__():
            raise RuntimeError(
                "The droplet index {} is out of bound".format(droplet_index))
        droplet = self.droplets[droplet_index]
        px, py = droplet.get_position()
        if action == 0:
            if droplet.type == 2:
                 droplet.move_reward(0)
            return -0.3 - 0.0001*self.step_count**1.5, constraint
        past_in_blocks = self.chip_state[px, py]


        if self.stall and droplet.finished:
            return 0.0, constraint

        prob = self.getMoveProb(px,py)
        occupied[px, py]=0
        if random.random() <= prob:
            Fail = False
            x, y = droplet.try2move(action) # try2move
            if x < 0 or x >= self.width or y < 0 or y>= self.length:
                Fail = True
            elif not np.all(occupied[max(0, x-1):min(occupied.shape[0], x+2), max(0, y-1):min(occupied.shape[1], y+2)] == 0):
                constraint = 1
                Fail = True
            else:
                cur_in = self.chip_state[x, y]
                if not past_in_blocks and cur_in:
                    Fail = True

            if Fail:
                reward = -0.5
                droplet.fail2movestep()
            else:
                reward = droplet.move_reward(action)
                if past_in_blocks and not cur_in:
                    reward += 0.5

        else:
            reward = -0.4
        droplet.last_action = action
        reward-= 0.0001*self.step_count**1.5
        nx, ny=droplet.get_position()
        occupied[px, py] = droplet_index+1
        occupied[nx, ny] = droplet_index+1
        return reward, constraint

    def getMoveProb(self, x,y):
        prob = self.chip.m_health[x][y]
        return prob

    def getTaskStatus(self):
        return [d.finished for d in self.droplets]

    def getglobalobs(self):
        '''
        format of gloabal observed
        3-l-w
        Droplets               in layer 0  [id 0 0] 
        Goal                   in layer 1  [x id 0]
        Obstacles              in layer 2  [0  0 1]
        '''
        def add_blocks_In_gloabal_Obs(g_obs):
            for m in self.chip.blocks:
                for x in range(m.x_min, m.x_max + 1):
                    for y in range(m.y_min, m.y_max + 1):
                        g_obs[0][x][y] = 1
            return g_obs

        def add_droplets_in_gloabal_Obs(g_obs):
            for i in range(self.droplets.__len__()):
                g_obs[1][tuple(self.droplets[i].pos)] = i+1
                #g_obs[1][self.droplets[i].des_x][self.droplets[i].des_y] = i+1
            return g_obs
        global_obs = np.zeros((2, self.width, self.length), dtype=np.int8) # ?
        global_obs = add_blocks_In_gloabal_Obs(global_obs)
        global_obs = add_droplets_in_gloabal_Obs(global_obs)
        self.global_obs = global_obs
        return global_obs

    def get_state_obs(self, last_actions=None, get_size=False):
        chip_state=np.zeros((self.width,self.length), dtype=np.int8)

        for m in self.chip.blocks:
            for x in range(m.x_min, m.x_max + 1):
                for y in range(m.y_min, m.y_max + 1):
                    chip_state[x][y] = 1
        size=[[5],[(2,9,9,5)],[(2,9,9,2,164)],[(3,9,9,2,245)],[(4,9,9,2,326)], [(4,9,9,2,326),(2,9,9,1,164)], [(4,9,9,2,326),(4,9,9,1,325)],[(4, self.length,self.width)],[(5, self.length,self.width)]]
        obs_set=[obs_1, obs_2, obs_3, obs_4, obs_5, obs2_0, obs2_1, obs3_0, obs3_1]

        if get_size: # 返回state的大小供replay_buffer初始化
            return size[self.oc]

        self.chip_state = chip_state

        self.last_frame=None
        if self.oc<7:
            observation = obs_set[self.oc](self.width, self.length, self.droplets, last_actions, self.fov[0])
            self.last_frame = np.zeros((self.width, self.length))
        elif self.oc==7:
            observation = obs_set[self.oc](self.width, self.length, self.droplets, self.chip.blocks)
            self.last_frame = observation[0]
        else: #oc==8
            observation = obs_set[self.oc](self.width, self.length, self.droplets, self.chip.blocks, self.last_frame)
            self.last_frame = observation[0]


        return observation

    def getOneObs(self, agent_i, type=0):
        '''
        format of gloabal observed
        3-l-w
        Droplets               in layer 0  [id 0 0 0]
        other's Goal           in layer 2  [x x id 0]
        Obstacles              in layer 3  [x 0  0 1]
        '''
        if agent_i is None:
            Drop=[Droplet_T1,Droplet_T2,Droplet_store]
            droplet=Drop[type]((0, 0))
        else:
            droplet=self.droplets[agent_i]


        fov = self.fov[droplet.type]  # 正方形fov边长
        hf=fov//2

        obs_i = np.zeros((2, fov, fov))
        center = np.array(droplet.pos) # 当前液滴所在坐标
        #

        # get block layer 0
        global_block=np.ones((self.width+fov, self.length+fov))
        global_block[hf:self.width+hf,hf:self.length+hf] = self.global_obs[0]  # 在原先的block周围加一圈宽为hf,值为1的boundary
        obs_i[0]=global_block[center[0]:center[0]+fov,center[1]:center[1]+fov] # 从中取fov所在区域
        global_drop = np.zeros((self.width + fov, self.length + fov))
        global_drop[hf:self.width + hf, hf:self.length + hf] = self.global_obs[1] # 在原先的droplet信息周围加一圈宽为hf,值为0的boundary
        obs_i[1] = global_drop[center[0]:center[0] + fov, center[1]:center[1] + fov]
  


        if droplet.type==0:
            # delete droplet on self goal if two drop come together
            if droplet.partner:
                # obs_i[0][obs_i[0] == droplet.partner] = 0
                obs_i[1][obs_i[1]==(self.droplets.index(droplet.partner)+1)]=0
            obs_i[1] = obs_i[1]>0
            dirct = droplet.direct_vector(self.width, self.length, hf)
            return obs_i, dirct

        # obs_i[1] = obs_i[1] > 0

        if droplet.type==1:
            if droplet.headto == 2:
                obs_i = np.flip(obs_i, 1)
            elif droplet.headto == 3:
                obs_i = np.rot90(obs_i, axes=(1, 2))
            elif droplet.headto == 4:
                obs_i = np.rot90(obs_i, axes=(2, 1))
            obs_i[1] = obs_i[1] > 0
            # return obs_i, np.array(droplet.mix_percent)
            return obs_i, np.array(droplet.mix_percent)

        return obs_i, None

    def get_avail_action(self, n, need=False):
        if not need:
            return [[1]*n]*self.droplets.__len__()
        avail_actions=[]
        i=0
        for d in self.droplets:
            arr=np.array([1]*n)
            x,y=d.get_position()
            if x==0:
                arr[2]=0
            elif self.global_obs[0,x-1,y]==1:
                arr[2] = 0
            if x == self.width-1:
                arr[1] = 0
            elif self.chip_state[x+1, y]==1:
                arr[1] = 0
            if y == 0:
                arr[3] = 0
            elif self.global_obs[0,x,y-1]==1:
                arr[3] = 0
            if y == self.length-1:
                arr[4] = 0
            elif self.global_obs[0,x,y+1]==1:
                arr[4] = 0
            if d.type==1:# 修改
                if d.headto==2:
                    arr[[0, 2, 1, 4, 3]] = arr[[0,1,2,3,4]]
                if d.headto==3:
                    arr[[0, 4,3,1,2]] = arr[[0,1,2,3,4]]
                if d.headto==4:
                    arr[[0, 3, 4, 2, 1]] = arr[[0,1,2,3,4]]
            avail_actions.append(arr)

        return avail_actions





class TrainingManager(RoutingTaskManager):
    def __init__(self, chip, task=-1,n_block=0, **kwargs):
        super().__init__(chip, **kwargs)
        droplet_limit = int((self.width+1)*(self.length+1)/10)

        self.n_droplets = 4
        self.GenDroplets = (self.GenDroplets_T1, self.GenDroplets_T2, self.GenDroplets_Store)
        self.task=task
        self.n_block = n_block

    @property
    def max_step(self):
        if self.task == 0:
            return (self.width + self.length) * 2
        if self.task == 1:
            return int(Droplet_T2.fullmix // 0.002)


    def Generate_task(self, drop_num):
        if not drop_num:
            raise TypeError('drop_num needed')
        self.chip.generate_random_chip(n_dis=0, n_block=self.n_block)
        if self.task == -1: # 每个任务各生成drop_num个，之后再改吧
            self.GenDroplets_T1(drop_num)
            self.GenDroplets_T2(drop_num)
            self.GenDroplets_Store()
        else:
            self.GenDroplets[self.task](drop_num)

    def GenDroplets_T1(self, drop_num):
        Start_End = self._Generate_Locations(2*drop_num)
        self.starts = Start_End[0:drop_num]
        self.ends = Start_End[drop_num:]
        for i in range(0, drop_num):
            self.droplets.add(0, self.starts[i], self.ends[i])
        self.distances = np.sum(np.abs(self.starts - self.ends), axis=1)
        self.max_dist= np.max(self.distances)

    def GenDroplets_T2(self, drop_num):
        starts = self._Generate_Locations(drop_num)
        for i in range(0, drop_num):
            self.droplets.add(1, starts[i], difficulty=0.5*drop_num+10-0.1*self.chip.width-0.1*self.chip.length)

    def GenDroplets_Store(self):
        pass

    def check_finish(self, step_count):
        terminated=False
        status = self.getTaskStatus()
        if np.all(status):
            terminated = True
        return terminated

    # reset the enviorment
    def refresh(self, drop_num=None):
        self.step_count = 0
        super().refresh()
        self.Generate_task(drop_num)

    def get_env_info(self):
        obs_shape=self.get_state_obs(get_size=True)
        print('obs shape: ', obs_shape)
        env_info = {"state_shape": obs_shape,
                    "obs_shape": obs_shape,
                    "episode_limit": self.max_step}
        return env_info

    def set4traintask2(self, revers=False):
        if  self.task == 1:
            Droplet_T2.fullmix=1
            if revers:
                Droplet_T2.fullmix=1




class AssayTaskManager(RoutingTaskManager):
    def __init__(self, chip, assay, **kwargs):
        super().__init__(chip, **kwargs)
        self.assay = assay
        self.task = assay.name

        Droplet_T2.fullmix = 1.0

    def check_finish(self, step_count):
        terminated = False
        self.predrop = [d.opid for d in self.droplets]
        self.assay.update_droplets(self.droplets, step_count,
                                                   self.chip)


        if self.droplets.__len__() == 0:
            terminated = True
        return terminated

    def refresh(self, drop_num=None):
        self.droplets.clear()
        self.assay.assign_ports(self.chip)
        self.assay.initial_candidate_list()
        self.assay.update_droplets(self.droplets, 0, self.chip)
        self.step_count = 0

    def get_env_info(self):
        state = self.getglobalobs().flatten()
        pixel0, vector0= self.getOneObs(None,0)
        pix1, vec1 = self.getOneObs(None,1)

        # print('obs shape: ', pixel.shape)
        env_info = {"state_shape": state.shape[-1],
                    "obs_shape": self.get_state_obs(get_size=True)
        } #(channel,fov,fov,vector lenth, whole size) whole size:obs_shape[-1]
        print('obs shape: ', env_info['obs_shape'])
        return env_info



class DMFBenv(ParallelEnv):
    """ A DMFB biochip environment
    [0,0]
        +---l---+-> x
        w       |
        +-------+
        |     [1,2]
        V
        y
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    # 环境初始化

    def __init__(self, routing_manager, show=False, savemp4=False):
        super(DMFBenv, self).__init__()
        # self.mode='human' if show else None
        self.actions = Action
        self.width, self.length = routing_manager.chip.size

        # self.n_agents = 0

        # # self.agents = ["player_{}".format(i) for i in range(n_agents)]
        # self.agents = ["player_{}".format(i) for i in range(droplets.__len__())]
        # # self.agents=list(range(n_agents))
        # self.possible_agents = self.agents[:]
        # self.action_spaces = {name: spaces.Discrete(len(self.actions))
        #                       for name in self.agents}
        # self.rewards = {i: 0. for i in self.agents}
        # self.dones = {i: False for i in self.agents}
        # # different
        # self.observation_spaces = {name: spaces.Box(
        #     low=0, high=len(droplets), shape=(3, self.width, self.length), dtype='uint8')
        #     for name in self.agents}

        # Other data members
        # self.routing_manager = RoutingTaskManager(
        #     chip, fov, stall, assay=task)
        self.routing_manager = routing_manager

        # variables below change every game
        self.step_count = 0
        self.constraints = 0

        self.mode=None
        if show or savemp4:
            self.mode = 'human'
            self.render_init(self.width, self.length, task=routing_manager.task, savemp4=savemp4)

    @property
    def n_agents(self):
        return len(self.routing_manager.droplets)


    def step(self, actions, record=True, timestep=None):

        success = 0
  
        shaped_weight = sparse_weight = 1.0

        if timestep is not None:
            decay_factor = min(1, timestep / 300000)
            shaped_weight = 1.0 - decay_factor
            sparse_weight = decay_factor

        acts = actions
        rewards, constraints, terminated = self.routing_manager.moveDroplets(acts)
        if record:
            self.routing_manager.chip.addUsage([d.pos for d in self.routing_manager.droplets if not d.finished])
        self.constraints += constraints

        obs = self.getObs([d.last_action for d in self.routing_manager.droplets])  # patitial observed consist of the Obs

        if terminated:
            rewards = [shaped_weight * r + sparse_weight * 10 for r in rewards]
            rewards = [rewards[i]+ sparse_weight * self.routing_manager.droplets[i].difficulty for i in range(self.n_agents)]
            success = 1
        # self.routing_manager.getglobalobs()  # update the state
        info = {'constraints': constraints, 'success': success}
        if terminated and self.mode is not None:
            print('constraints:', self.constraints)

        return obs, rewards, terminated, info

    def reset_chip(self, w, l):
        self.routing_manager.chip= Chip(w, l)
        self.routing_manager.width = w
        self.routing_manager.length = l
        self.width, self.length = (w,l)


    def reset(self, n_agents=None, evaluate=False):
        self.constraints = 0
        self.routing_manager.refresh(drop_num=n_agents) # 会更新degrade, 是new就是新生成的
        obs = self.getObs()
        self.render(refresh=True)
        return obs

    def restart(self, index=None):
        self.routing_manager.restartforall()
        self.step_count = 0
        self.constraints = 0
        return self.getObs()

    def seed(self, seed=None):
        pass


    def getObs(self, actions=None):  # partitial observertion for all droplets
        last_actions = np.zeros((self.n_agents, len(self.actions)))
        if actions is not None:
            i=0
            for idx in actions:
                last_actions[i,idx] = 1
                i += 1
        return self.routing_manager.get_state_obs(last_actions)



    def get_env_info(self):
        env_info=self.routing_manager.get_env_info()
        env_info["n_actions"]=len(self.actions)
        return env_info




    def render(self, refresh=False, close=False):
        if self.mode is None:
            return

        def is_goal(position,droplets):
            for i in range(len(droplets)):
                if np.array_equal(position, self.routing_manager.ends[i]):
                    return i
            return False

        def is_out_edge(position):
            if 2 > position[0] or position[0] > self.length + 1 or 2 > position[1] or position[1] > self.width + 1:
                return True
            return False

        droplets = self.routing_manager.droplets

        def drawcell(u_size, dgrey=255):
            cell = np.ones((u_size, u_size, 3), dtype='uint8') * dgrey
            cell[:, [0, -2, -1], :] = 0
            cell[[0, -2, -1], :, :] = 0
            return cell


        # used for render
        self.agent_redender = [None]*len(self.routing_manager.droplets)
        self.agenttrans = [None]*len(self.routing_manager.droplets)

        import pygame
        from PIL import Image

        if close:
            if self.viewer is not None:
                pygame.display.set_mode((800, 800))
                pygame.display.flip()
                # pygame.quit()
                self.viewer = None
            if self.video is not None:
                self.video.release()
            return None

        u_size = self.u_size  # cell 尺寸
        m = 1  # cell 间隔
        screen_length = self.screen_length
        screen_width = self.screen_width
        if self.viewer is None:

            pygame.init()
            self.viewer = pygame.display.set_mode(
                (screen_width, screen_length))  # rendering.Viewer(self.env_length, self.env_width)
        # 背景
        if refresh:
            background = pygame.Surface((screen_width, screen_length))
            background.fill('grey')
            blockarr = drawcell(u_size - 2 * m, 10)
            block = pygame.image.frombuffer(blockarr.flatten(), (u_size - 2 * m, u_size - 2 * m), 'RGB').convert()
            for x in range(1, self.width+1):
                for y in range(1, self.length+1):
                    if self.routing_manager.chip.grid[x - 1, y - 1] == 0:
                        background.blit(block, (x * u_size + m, y * u_size + m))  # block
                    else:
                        cellarr = drawcell(u_size - 2 * m, 100+int(155*self.routing_manager.chip.m_health[x-1][y-1]))
                        cell = pygame.image.frombuffer(cellarr.flatten(), (u_size - 2 * m, u_size - 2 * m),
                                                       'RGB').convert()
                        background.blit(cell, (x * u_size + m, y * u_size + m))  # 普通格子
            # # dispense
            icon = pygame.image.frombuffer(
                Image.open('../fig/dispense_port.bmp').resize((u_size, u_size)).tobytes(), (u_size, u_size), 'RGB') \
                .convert()
            fonta = pygame.font.SysFont("Times New Roman", 30)
            for port in self.routing_manager.chip.ports:
                # print(port[0], port[1])
                direction = ["→", "↓", "←", "↑"]
                arrow = fonta.render(direction[port.direction], True, (0, 0, 0), )  # (255, 255, 255))
                background.blit(icon, ((port.drawpos[0]) * u_size, (port.drawpos[1]) * u_size))
                background.blit(arrow, ((port.drawpos[0]) * u_size+u_size/5, (port.drawpos[1]) * u_size+u_size/7))

            self.background = background

        self.viewer.blit(self.background, (0, 0))
        unused = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        for d in droplets:
            if hasattr(d,'name'):
                unused.remove(d.name)
        font = pygame.font.SysFont("Times New Roman", 15)
        for d in droplets:
            if not hasattr(d,'dropicon'):
                idx = unused.pop(0)
                setattr(d,'dropicon',pygame.image.frombuffer(
                Image.open('../fig/droplet{}.png'.format(idx)).resize((u_size, u_size)).tobytes(), (u_size, u_size),
                'RGBA').convert_alpha())
                setattr(d,'name',idx)
            x, y = d.pos
            x+=1
            y+=1
            self.viewer.blit(d.dropicon, (x * u_size, y * u_size))
            if d.type == 0:
                if not hasattr(d, 'goalicon'):
                    goalfile = '../fig/goal{}.png'.format(d.name)
                    goal = pygame.image.load(goalfile).convert_alpha()
                    goal = pygame.transform.scale(goal, (u_size - m, u_size - m))
                    setattr(d,'goalicon',goal)
                self.viewer.blit(d.goalicon, ((d.des[0]+1) * u_size, (d.des[1]+1) * u_size))
            # 显示百分比

            if d.type == 1:
                text = font.render("%.1f" % (d.mix_percent * 100 /d.fullmix) + "%", True, (0, 0, 0), )  # (255, 255, 255))
                self.viewer.blit(text, ((d.pos[0] + 1) * u_size, (d.pos[1] + 1.25) * u_size))
                pass


        if self.save:
            imagestring = pygame.image.tostring(self.viewer.subsurface(0, 0, screen_width, screen_length), "RGB")
            pilImage = Image.frombytes("RGB", (screen_width, screen_length), imagestring)
            imag = cv2.cvtColor(np.asarray(pilImage), cv2.COLOR_RGB2BGR)
            self.video.write(imag)
        time.sleep(0.01)
        return pygame.display.flip()  # if mode == 'human' else img

    def render_init(self,width,length, task, savemp4=False):
        
        self.u_size = 40  # size for cell pixels
        if length > 30:
            self.u_size = 20
        self.viewer = None
        self.save = False
        self.screen_length = self.u_size * (length+2)
        self.screen_width = self.u_size * (width+2)
        self.video = None
        if savemp4:
            self.save = True
            file_path = 'video/{}by{}-T{}'.format(
                width, length, task) + str(
                int(time.time())) + ".avi"  # 导出路径

            fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')  # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）
            fps = 12
            self.video = cv2.VideoWriter(file_path, fourcc, fps, (self.screen_width, self.screen_length))
            print('到这了么')

    def close(self):
        try:
            self.viewer
        except AttributeError:
            return
        if self.viewer:
            self.render(close=True)

def obs_1(w,l, droplets, last_actions, fov):
    n = droplets.__len__()
    points = np.zeros((n, 2))
    partners = np.zeros((n,), dtype=int)
    drop = np.zeros((n, 5))
    for i, d in enumerate(droplets):
        if d.type == 0:
            if d.partner:
                partners[i] = droplets.index(d.partner)
            else:
                partners[i] = -1
            points[i] = d.pos
            drop[i] = (d.type, d.pos[0], d.pos[1], d.des[0], d.des[1])
        elif d.type == 1:
            drop[i] = (d.type, d.pos[0], d.pos[1], d.mix_percent, d.headto)
        else:  # store液滴先跟type0共用策略
            drop[i] = (d.type, d.pos[0], d.pos[1], d.pos[0], d.pos[1])

    dis = euclidean_distance_matrix(points)
    dis[dis > fov] = fov + 1
    scaled_d = 1 - (dis / (fov + 1))
    scaled_d = scaled_d / np.sum(scaled_d, axis=-1, keepdims=True)

    indices = np.where(partners != -1)
    scaled_d[indices, partners[indices]] = 0

    drop = np.concatenate([drop, last_actions], axis=-1)

    drop = np.concatenate([drop, scaled_d], axis=-1)
    return drop

def obs_2(w,l, droplets, last_actions, fov):
    gobs=np.zeros((w,l))
    n = droplets.__len__()
    points = np.zeros((n, 2))
    partners = np.zeros((n,), dtype=int)
    drop = np.zeros((n, 5))
    for i, d in enumerate(droplets):
        gobs[tuple(d.pos)] = i + 1
        if d.type == 0:
            if d.partner:
                partners[i] = droplets.index(d.partner)
            else:
                partners[i] = -1
            points[i] = d.pos
            drop[i] = (d.type, d.pos[0], d.pos[1], d.des[0], d.des[1])
        elif d.type == 1:
            drop[i] = (d.type, d.pos[0], d.pos[1], d.mix_percent, d.headto)
        else:  # store液滴先跟type0共用策略
            drop[i] = (d.type, d.pos[0], d.pos[1], d.pos[0], d.pos[1])

    dis = euclidean_distance_matrix(points)
    dis[dis > fov] = fov + 1
    scaled_d = 1 - (dis / (fov + 1))
    scaled_d = scaled_d / np.sum(scaled_d, axis=-1, keepdims=True)

    indices = np.where(partners != -1)
    scaled_d[indices, partners[indices]] = 0

    drop = np.concatenate([drop, last_actions], axis=-1)

    drop = np.concatenate([drop, scaled_d], axis=-1)

    # 把以前的贴过来
    obs=np.zeros((n,2,fov,fov))
    global_block = np.ones((w + fov, l + fov))
    hf = fov //2
    global_block[hf:w+ hf, hf:l + hf] = np.zeros((w,l))  # 在原先的block周围加一圈宽为hf,值为1的boundary
    global_drop = np.zeros((w + fov, l + fov))
    global_drop[hf:w + hf, hf:l + hf] = gobs  # 在原先的droplet信息周围加一圈宽为hf,值为0的boundary
    for i, d in enumerate(droplets):
        center = np.array(d.pos)  # 当前液滴所在坐标
        obs[i, 0] = global_block[center[0]:center[0] + fov, center[1]:center[1] + fov]  # 从中取fov所在区域
        obs[i,1] = global_drop[center[0]:center[0] + fov, center[1]:center[1] + fov]
 
    obs_flat = obs.reshape((n, -1))

    return np.concatenate([obs_flat, drop], axis=-1)

def obs_3(w,l, droplets, last_actions, fov):
    gobs=np.zeros((w,l))
    n = droplets.__len__()
    points = np.zeros((n, 2))
    partners = np.zeros((n,), dtype=int)
    drop = np.zeros((n, 5))
    for i, d in enumerate(droplets):
        gobs[tuple(d.pos)] = i + 1
        if d.type == 0:
            if d.partner:
                partners[i] = droplets.index(d.partner)
            else:
                partners[i] = -1
            points[i] = d.pos
            drop[i] = (d.type, d.pos[0], d.pos[1], d.des[0], d.des[1])
        elif d.type == 1:
            drop[i] = (d.type, d.pos[0], d.pos[1], d.mix_percent, d.headto)
        else:  # store液滴先跟type0共用策略
            drop[i] = (d.type, d.pos[0], d.pos[1], d.pos[0], d.pos[1])

    dis = euclidean_distance_matrix(points)
    dis[dis > fov] = fov + 1
    scaled_d = 1 - (dis / (fov + 1))
    scaled_d = scaled_d / np.sum(scaled_d, axis=-1, keepdims=True)

    # 删除partner
    indices = np.where(partners != -1)
    scaled_d[indices, partners[indices]] = 0

    # 拼last action
    drop = np.concatenate([drop, last_actions], axis=-1)

    drop = np.concatenate([drop, scaled_d], axis=-1)

    # 把以前的贴过来
    obs=np.zeros((n,2,fov,fov))
    global_block = np.ones((w + fov, l + fov))
    hf = fov //2
    global_block[hf:w+ hf, hf:l + hf] = np.zeros((w,l))  # 在原先的block周围加一圈宽为hf,值为1的boundary
    global_drop = np.zeros((w + fov, l + fov))
    global_drop[hf:w + hf, hf:l + hf] = gobs  # 在原先的droplet信息周围加一圈宽为hf,值为0的boundary
    dirct=np.zeros((n,2))
    for i, d in enumerate(droplets):
        center = np.array(d.pos)  # 当前液滴所在坐标
        obs[i, 0] = global_block[center[0]:center[0] + fov, center[1]:center[1] + fov]  # 从中取fov所在区域
        obs[i, 1] = global_drop[center[0]:center[0] + fov, center[1]:center[1] + fov]

        dirct[i] = d.direct_vector(w, l, hf)
    obs[:,1] = obs[:,1] > 0
    obs_flat = obs.reshape((n, -1))
    dirct = np.concatenate([dirct, last_actions], axis=-1)

    return np.concatenate([obs_flat, dirct], axis=-1)

def obs_4(w,l, droplets, last_actions, fov):
    gobs=np.zeros((3,w,l))
    n = droplets.__len__()
    points = np.zeros((n, 2))
    partners = np.zeros((n,), dtype=int)
    drop = np.zeros((n, 5))
    dirct = np.zeros((n, 2))
    hf = fov // 2
    for i, d in enumerate(droplets):
        gobs[(0,)+tuple(d.pos)] = i + 1
        dirct[i] = d.direct_vector(w, l, hf)
        gobs[(1,)+tuple(d.pos)] = dirct[i][0]
        gobs[(2,)+tuple(d.pos)] = dirct[i][1]
        if d.type == 0:
            if d.partner:
                partners[i] = droplets.index(d.partner)
            else:
                partners[i] = -1
            points[i] = d.pos
            drop[i] = (d.type, d.pos[0], d.pos[1], d.des[0], d.des[1])
        elif d.type == 1:
            drop[i] = (d.type, d.pos[0], d.pos[1], d.mix_percent, d.headto)
        else:  # store液滴先跟type0共用策略
            drop[i] = (d.type, d.pos[0], d.pos[1], d.pos[0], d.pos[1])

    dis = euclidean_distance_matrix(points)
    dis[dis > fov] = fov + 1
    scaled_d = 1 - (dis / (fov + 1))
    scaled_d = scaled_d / np.sum(scaled_d, axis=-1, keepdims=True)

    # 删除partner
    indices = np.where(partners != -1)
    scaled_d[indices, partners[indices]] = 0

    # 把以前的贴过来
    obs=np.zeros((n,3,fov,fov))
    global_block = np.ones((w + fov, l + fov))

    global_block[hf:w+ hf, hf:l + hf] = np.zeros((w,l))  # 在原先的block周围加一圈宽为hf,值为1的boundary
    global_drop = np.zeros((3, w + fov, l + fov))
    global_drop[:, hf:w + hf, hf:l + hf] = gobs  # 在原先的droplet信息周围加一圈宽为hf,值为0的boundary
    # 增加一圈？

    for i, d in enumerate(droplets):
        center = np.array(d.pos)  # 当前液滴所在坐标

        # get block layer 0
        obs[i, 0] = global_block[center[0]:center[0] + fov, center[1]:center[1] + fov]  # 从中取fov所在区域

        # get droplet layer 1
        drop_temp = global_drop[0, center[0]:center[0] + fov, center[1]:center[1] + fov].copy()
        drop_temp[drop_temp == i + 1] = 0
        if d.type == 0:
            if d.partner:
                drop_temp[drop_temp == (d.partner.id + 1)] = 0
        drop_temp = cv2.dilate(drop_temp, np.ones((3, 3)))

        obs[i, 0] = obs[i, 0] + drop_temp
        obs[i, 1:] = global_drop[1:, center[0]:center[0] + fov, center[1]:center[1] + fov]

        # if droplet.type == 0:
        #     if droplet.partner:
        #         obs_i[1][obs_i[1] == (self.droplets.index(droplet.partner) + 1)] = 0
        #     obs_i[1] = obs_i[1] > 0
    obs[:,0] = obs[:,0]>0
    obs_flat = obs.reshape((n, -1))
    dirct = np.concatenate([dirct, last_actions], axis=-1)

    return np.concatenate([obs_flat, dirct], axis=-1)

def obs_5(w,l, droplets, last_actions, fov):
    gobs=np.zeros((3,w,l))
    n = droplets.__len__()
    dirct = np.zeros((n, 2))
    hf = fov // 2
    for i, d in enumerate(droplets):
        gobs[(0,)+tuple(d.pos)] = i + 1
        dirct[i] = d.direct_vector(w, l, hf)
        gobs[(1,)+tuple(d.pos)] = dirct[i][0]
        gobs[(2,)+tuple(d.pos)] = dirct[i][1]



    # 把以前的贴过来
    obs=np.zeros((n,4,fov,fov))
    global_block = np.ones((w + fov, l + fov))

    global_block[hf:w+ hf, hf:l + hf] = np.zeros((w,l))  # 在原先的block周围加一圈宽为hf,值为1的boundary
    global_drop = np.zeros((3, w + fov, l + fov))
    global_drop[:, hf:w + hf, hf:l + hf] = gobs  # 在原先的droplet信息周围加一圈宽为hf,值为0的boundary
    # 增加一圈？

    for i, d in enumerate(droplets):
        center = np.array(d.pos)  # 当前液滴所在坐标

        # get block layer 0
        obs[i, 0] = global_block[center[0]:center[0] + fov, center[1]:center[1] + fov]  # 从中取fov所在区域

        # get droplet layer 1
        drop_temp = global_drop[0, center[0]:center[0] + fov, center[1]:center[1] + fov].copy()
        if d.type == 0:
            if d.partner:
                drop_temp[drop_temp == (d.partner.id + 1)] = 0
        obs[i, 1] = drop_temp
        drop_temp[drop_temp == i + 1] = 0
        drop_temp = cv2.dilate(drop_temp, np.ones((3, 3)))

        obs[i, 0] = obs[i, 0] + drop_temp
        obs[i, 2:] = global_drop[1:, center[0]:center[0] + fov, center[1]:center[1] + fov]

    obs[:, 0] = obs[:, 0]>0
    obs[:, 1] = obs[:, 1] > 0
    obs_flat = obs.reshape((n, -1))
    dirct = np.concatenate([dirct, last_actions], axis=-1)

    return np.concatenate([obs_flat, dirct], axis=-1)

def obs2_0(w,l, droplets, last_actions, fov):
    # combine obs_5 for type1
    gobs = np.zeros((3, w, l))
    n = droplets.__len__()
    dirct_type0 = np.zeros((n, 2))
    dirct_type1 = np.zeros(n)
    hf = fov // 2
    for i, d in enumerate(droplets):
        gobs[(0,) + tuple(d.pos)] = i + 1
        if d.type == 1:
            dirct_type1[i] = d.mix_percent
        dirct_type0[i] = d.direct_vector(w, l, hf)
        gobs[(1,) + tuple(d.pos)] = dirct_type0[i][0]
        gobs[(2,) + tuple(d.pos)] = dirct_type0[i][1]

    # 把以前的贴过来
    obs = np.zeros((n, 4, fov, fov))
    global_block = np.ones((w + fov, l + fov))

    global_block[hf:w + hf, hf:l + hf] = np.zeros((w, l))  # 在原先的block周围加一圈宽为hf,值为1的boundary
    global_drop = np.zeros((3, w + fov, l + fov))
    global_drop[:, hf:w + hf, hf:l + hf] = gobs  # 在原先的droplet信息周围加一圈宽为hf,值为0的boundary
    # 增加一圈？

    obs = []
    for i, d in enumerate(droplets):
        center = np.array(d.pos)  # 当前液滴所在坐标

        # get block layer 0
        obs_temp = np.zeros((4, fov, fov))
        obs_temp[0] = global_block[center[0]:center[0] + fov, center[1]:center[1] + fov]  # 从中取fov所在区域

        # get droplet layer 1
        drop_temp = global_drop[0, center[0]:center[0] + fov, center[1]:center[1] + fov].copy()
        if d.type == 0:
            if d.partner:
                drop_temp[drop_temp == (droplets.index(d.partner) + 1)] = 0
        obs_temp[1] = drop_temp
        drop_temp[drop_temp == i + 1] = 0
        drop_temp = cv2.dilate(drop_temp, np.ones((3, 3)))

        obs_temp[0] = obs_temp[0] + drop_temp
        obs_temp[2:] = global_drop[1:, center[0]:center[0] + fov, center[1]:center[1] + fov]

        obs_temp[0] = obs_temp[0] > 0
        obs_temp[1] = obs_temp[1] > 0
        if d.type == 1:
            if obs_temp is None:
                print('?')
            obs_temp = rotate(obs_temp, d.headto)
            if obs_temp is None:
                print('?')
            obs.append(np.append(obs_temp[:2], np.append(dirct_type1[i], last_actions[i])))
        elif d.type == 0:
            obs.append(np.append(obs_temp, np.concatenate([dirct_type0[i], last_actions[i]])))
        else:
            obs.append(0)

    try:
        obs = np.array(obs)
        return obs
    except:
        return obs

def obs2_1(w,l, droplets, last_actions, fov):
    # combine obs_5 for type1
    gobs = np.zeros((3, w, l))
    n = droplets.__len__()
    dirct_type0 = np.zeros((n, 2))
    dirct_type1 = np.zeros(n)
    hf = fov // 2
    for i, d in enumerate(droplets):
        gobs[(0,) + tuple(d.pos)] = i + 1
        if d.type == 1:
            dirct_type1[i] = d.mix_percent
        dirct_type0[i] = d.direct_vector(w, l, hf)
        gobs[(1,) + tuple(d.pos)] = dirct_type0[i][0]
        gobs[(2,) + tuple(d.pos)] = dirct_type0[i][1]

    # 把以前的贴过来
    # obs = np.zeros((n, 4, fov, fov))
    global_block = np.ones((w + fov, l + fov))

    global_block[hf:w + hf, hf:l + hf] = np.zeros((w, l))  # 在原先的block周围加一圈宽为hf,值为1的boundary
    global_drop = np.zeros((3, w + fov, l + fov))
    global_drop[:, hf:w + hf, hf:l + hf] = gobs  # 在原先的droplet信息周围加一圈宽为hf,值为0的boundary
    # 增加一圈？

    obs = []
    for i, d in enumerate(droplets):
        center = np.array(d.pos)  # 当前液滴所在坐标

        # get block layer 0
        obs_temp = np.zeros((4, fov, fov))
        obs_temp[0] = global_block[center[0]:center[0] + fov, center[1]:center[1] + fov]  # 从中取fov所在区域

        # get droplet layer 1
        drop_temp = global_drop[0, center[0]:center[0] + fov, center[1]:center[1] + fov].copy()
        if d.type == 0:
            if d.partner:
                drop_temp[drop_temp == (droplets.index(d.partner) + 1)] = 0
        obs_temp[1] = drop_temp
        drop_temp[drop_temp == i + 1] = 0
        drop_temp = cv2.dilate(drop_temp, np.ones((3, 3)))

        obs_temp[0] = obs_temp[0] + drop_temp
        obs_temp[2:] = global_drop[1:, center[0]:center[0] + fov, center[1]:center[1] + fov]

        obs_temp[0] = obs_temp[0] > 0
        obs_temp[1] = obs_temp[1] > 0
        if d.type == 1:
            if obs_temp is None:
                print('?')
            obs_temp = rotate(obs_temp, d.headto)
            if obs_temp is None:
                print('?')
            obs.append(np.append(obs_temp, np.append(dirct_type1[i], last_actions[i])))
        elif d.type == 0:
            obs.append(np.append(obs_temp, np.concatenate([dirct_type0[i], last_actions[i]])))
        else:
            obs.append(0)
    try:
        obs = np.array(obs)
        return obs
    except:
        return obs

def obs3_0(w, l, droplets, blocks):
    gobs=np.zeros((4,w,l))
    n = droplets.__len__()
    dirct = np.zeros((n, 2))
    for i, d in enumerate(droplets):
        gobs[(0,)+tuple(d.pos)] = i + 1
        dirct[i] = d.direct_vector(w, l, 3)
        gobs[(2,)+tuple(d.pos)] = dirct[i][0]
        gobs[(3,)+tuple(d.pos)] = dirct[i][1]
    for m in blocks:
        for x in range(m.x_min, m.x_max + 1):
            for y in range(m.y_min, m.y_max + 1):
                gobs[1][x][y] = 1
    return gobs

def obs3_1(w, l, droplets, blocks, last_frame): #增加一格上一帧
    gobs=np.zeros((5, w, l))
    gobs[1]=np.ones((w,l))
    n = droplets.__len__()
    dirct = np.zeros((n, 2))
    for i, d in enumerate(droplets):
        gobs[(0,)+tuple(d.pos)] = i + 1 # droplets
        dirct[i] = d.direct_vector(w, l, 3)
        # gobs[1, max(0, d.pos[0]-1):min(w, d.pos[0]+2), max(0, d.pos[1]-1):min(l, d.pos[1]+2)] = 0
        gobs[(2,)+tuple(d.pos)] = dirct[i][0]
        gobs[(3,)+tuple(d.pos)] = dirct[i][1]
    for m in blocks:
        for x in range(m.x_min, m.x_max + 1):
            for y in range(m.y_min, m.y_max + 1):
                gobs[1][x][y] = 0
    if last_frame:
        gobs[-1] = last_frame
    else:
        gobs[-1] = gobs[0]
    return gobs

from .observation import guide_line
def obs3_2(w, l, droplets, blocks, last_frame): #再加上指引 （前3个点）
    gobs=np.zeros((5, w, l))
    gobs[1]=np.ones((w,l)) # 1能走，0不能走？
    n = droplets.__len__()
    dirct = np.zeros((n, 2))
    guide_lines = np.zeros((n, 3, 2))
    for i, d in enumerate(droplets):
        gobs[(0,)+tuple(d.pos)] = i + 1 # droplets
        dirct[i] = d.direct_vector(w, l, 3)
        # gobs[1, max(0, d.pos[0]-1):min(w, d.pos[0]+2), max(0, d.pos[1]-1):min(l, d.pos[1]+2)] = 0 #这是啥意思？ 哦把所有液滴设为block
        gobs[(2,)+tuple(d.pos)] = dirct[i][0]
        gobs[(3,)+tuple(d.pos)] = dirct[i][1]
        guide_lines[i]=np.array(guide_line(d, blocks))
    for m in blocks:
        for x in range(m.x_min, m.x_max + 1):
            for y in range(m.y_min, m.y_max + 1):
                gobs[1][x][y] = 0
    if last_frame:
        gobs[-1] = last_frame
    else:
        gobs[-1] = gobs[0]

    return gobs


def rotate(obs, dirct):

    if dirct == 2:
        return np.flip(obs, 1)
    elif dirct == 3:
        return np.rot90(obs, axes=(1, 2))
    elif dirct == 4:
        return np.rot90(obs, axes=(2, 1))
    else:
        return obs

if  __name__ == '__main__':
    manager=TrainingManager(Chip(10, 10, n_block=1), fov=[9,9])
    a=manager.get_state(True)
    print(a[0], a[1], a[-1])
    '''(10, 10) 4 False'''
    env=DMFBenv(manager)
    obs = env.reset(4)
    print(obs,obs[1].shape)
    '''.
    (array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int8), array([[ 1.,  9.,  7.,  8.],
       [ 6.,  5.,  1.,  4.],
       [ 4.,  0.,  3.,  6.],
       [ 3.,  8.,  0.,  0.],
       [ 7.,  6.,  0., -1.],
       [ 2.,  9.,  0., -1.],
       [ 4.,  5.,  0., -1.],
       [ 9.,  8.,  0., -1.]]))(8,4)
    '''


