import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def PCR(y=None):
    G = nx.DiGraph()
    for i in range(8):
        G.add_node(i, htype='ds', type='ds', pos=(i + 1, 0), dur=100, finished=False, started=False)
    for i in range(4):
        G.add_node(i + 8, htype='conf', type='mix', pos=[0, 1], finished=False)
        G.add_edges_from([(2 * i, i + 8), (2 * i + 1, i + 8)])
    G.add_node(12, htype='conf', type='mix', pos=[0, 2], finished=False)
    G.add_node(13, htype='conf', type='mix', pos=[0, 2], finished=False)
    G.add_node(14, htype='conf', type='mix', pos=[0, 3], finished=False)
    G.add_edges_from([(8, 12), (9, 12), (10, 13), (11, 13), (12, 14), (13, 14)])
    G.name = 'PCR'
    G.porttyp = {'ds': 8}
    return G

def in_vitro(inp=(2,3)):
    p, q = inp
    G = nx.DiGraph()
    pin = 0
    qjn = p * q
    m = 2 * p * q
    d = 3 * p * q
    dss = ['S1', 'S2', 'S3', 'S4']
    dsr = ['R1', 'R2', 'R3', 'R4']
    mix = ['M1', 'M2', 'M3', 'M4']
    det = ['D1', 'D2', 'D3', 'D4']
    for i in range(p):
        for j in range(q):
            G.add_node(pin, type=dss[i], htype='ds', dur=100)
            G.add_node(qjn, type=dsr[j], htype='ds', dur=100)
            G.add_node(m, htype='conf', type=mix[i])
            G.add_node(d, htype='nonc', type=det[j], dur=300)
            G.add_edges_from([(pin, m), (qjn, m), (m, d)])
            pin += 1
            qjn += 1
            m += 1
            d += 1
    # G.porttyp = True
    G.porttyp = {'S1':1,'S2':1,'S3':1,'S4':1,'R1':1,'R2':1,'R3':1,'R4':1}
    print('invitro' + str(inp))
    return G

def create_dispense(width, length, n):
    arr = np.zeros((width, length))
    boundary = [(0, i) for i in range(length)] + [(i, 0) for i in range(1, width)] + [(width - 1, i) for i in
                                                                                      range(length)] + [(i, length - 1)
                                                                                                        for i in
                                                                                                        range(1,
                                                                                                              width - 1)]
    chosen = []
    while len(chosen) < n:
        pos = np.random.choice(len(boundary))
        if all(abs(boundary[pos][0] - c[0]) + abs(boundary[pos][1] - c[1]) > 2 or (
                abs(boundary[pos][0] - c[0]) != abs(boundary[pos][1] - c[1]) and abs(boundary[pos][0] - c[0]) + abs(
            boundary[pos][1] - c[1]) == 2) for c in chosen):
            chosen.append(boundary.pop(pos))
    for pos in chosen:
        arr[pos] = -1
    return arr, chosen


mix_p = [0.29, 0.58, 0.1, -0.5]




class DAG(nx.DiGraph):
    def __init__(self, G):
        super().__init__(G)
        self.dispensers = [node for node in G.nodes() if G.in_degree(node) == 0]
        self.copy = G
        self.noconfig = [node for node in G.nodes() if G.nodes[node]['htype'] == 'noc'] + self.dispensers
        nodes = list(self.nodes())
        self.m = 1
        for node in nodes:
            if self.out_degree(node) == 0 or (self.out_degree(node)==1 and self.nodes[node]['type'] == 'dlt'):
                new_node_id = self.number_of_nodes()
                self.add_node(new_node_id, htype='out', type='out')
                self.add_edge(node, new_node_id)
        self.initial_candidate_list()

    def assign_ports(self, chip):
        if self.copy.porttyp:
            portype=self.copy.porttyp
            nports=sum(portype.values())
            chip.create_dispense(nports+1)
            tports={}
            i=0
            for type in portype.keys():
                tports[type] = ([],[])
                for _ in range(portype[type]):
                    tports[type][0].append(chip.ports[i])
                    i+=1
                    tports[type][0][-1].dur=50
            for node in self.dispensers:
                tports[self.nodes[node]['type']][1].append(node)
            self.tports = tports
        else:
            chip.create_dispense(len(self.dispensers) + 1)
            i = 0
            for node in self.dispensers:
                self.nodes[node]['pos'] = chip.ports[i].pos
                chip.ports[i].dur = 100
                i += 1
        self.out = chip.ports[-1].position
        chip.ports[-1].direction -= 2 # 画画时箭头反向


    def _add_trans_nodes(self, out):
        edges_to_remove = []
        for u, v in self.edges():
            new_node_id = self.number_of_nodes()
            self.add_node(new_node_id, htype='T1', type='trans', pos=list(u.pos))
            self.add_edge(u, new_node_id)
            self.add_edge(new_node_id, v)
            edges_to_remove.append((u, v))
        for u, v in edges_to_remove:
            self.remove_edge(u, v)
        for node in self.nodes():
            if self.out_degree(node) == 0:
                new_node_id = self.number_of_nodes()
                self.add_node(new_node_id, htype='T1', type='out', pos=out)
                self.add_edge(node, new_node_id)

    def initial_candidate_list(self):
        self.non_dis=self.copy.copy()
        self.non_dis.remove_nodes_from(self.dispensers)
        G = self.non_dis
        self.CL = [node for node in G.nodes() if G.in_degree(node) == 0]
        return

    def check_op2start(self, i, chip, drops):
        G = self
        m = self.m -1
        flag = [True]*len(list(G.predecessors(i)))
        dispenspaires= {}
        j = -1
        for father in G.predecessors(i):
            j += 1
            fop = G.nodes[father]
            if fop['htype'] == 'ds':
                flag[j] = False
                ports, ops = self.tports[fop['type']]
                for port in ports:
                    if port.occupied and port.occupied.time >= port.dur:
                        if (port, ops) in dispenspaires.values():
                            continue
                        dispenspaires[father]=(port, ops)
                        flag[j] = True
                        break
        if min(flag):
            for father, (port, ops) in dispenspaires.items():
                fop = G.nodes[father]
                fop['pos'] = port.position
                ops.remove(father)
                drops.remove(port.occupied)
                port.occupied = False
        
        if min(flag):
            for son in G.successors(i):
                son_type = G.nodes[son]
                if son_type['type'] == 'out' or son_type['htype'] == 'nonc':
                    continue
                else:
                    m += 1
            if m > 6:
                flag = False
            

        return min(flag)
    

    def dispense_droplets(self, droplets):
        for type in self.tports.keys():
            ports, ops=self.tports[type]
            for port in ports:
                if not port.occupied and len(ops) >= len(ports):
                    droplets.add(2, port.position, resource=port)

    def check_CL(self, droplets, chip):
        CL = self.CL
        G = self
        self.dispense_droplets(droplets)
        for i in CL[:]:  # [:]防止删一个跳一个
            if self.check_op2start(i, chip, droplets):
                op = G.nodes[i]
                CL.remove(i)
                # 混合or 稀释操作
                if op['htype'] == 'conf':
                    d = []
                    for father in G.predecessors(i):  # 两个父节点
                        fop = G.nodes[father]
                        # 非分配操作一定产生 store，父节点记录其完成后store的液滴
                        if 'store' in fop:
                            store = fop['store']
                            if store.ref > 1:
                                d.append(store.pos.copy())
                            else:
                                d.append(store.pos)
                            store.ref -= 1
                            if store.ref <= 0:
                                droplets.remove(store)
                                del fop['store']
                        # 说明是分配操作
                        else:
                            d.append(list(fop['pos']))
                    print(len(d))
                    droplets.addcp(d[0], d[1], opid=i)
                elif op['htype'] == 'nonc':
                    father = list(G.predecessors(i))[0]  # 只有一个父节点
                    fop = G.nodes[father]
                  
                    if 'store' in fop:
                        store = fop['store']
                        store.duration = op['dur']
                        store.opid = i
                        op['store'] = store
                        del fop['store']

    def update_droplets(self, droplets, t, chip):
        G = self
        CL = self.CL
        ned_remove = []
        for d in droplets:
            if d.finished:
                if d.opid == 80:
                    print('stop')
                ned_remove.append(d)
                if d.opid != 0: # 输出和 store opid=0
                    op = G.nodes[d.opid]
                    if d.type == 0: # 只是传输过程，操作并没有完成
                        if op['htype'] == 'conf':
                            droplets.add(1, d.pos.copy(), opid=d.opid)
                            if d.partner:
                                droplets.remove(d.partner)
                        elif op['htype'] == 'nonc':
                            op['stop'] = t + op['dur']
                    else:
                        if d.opid in G.non_dis:
                            G.non_dis.remove_node(d.opid)
                            if 'store' in op: #他本身就是resource op, 不可能有多个后继
                                store = op['store']
                            else:
                                store = droplets.add(2, d.pos)
                                op['store'] = store
                            for s in G.successors(d.opid):
                                store.ref += 1 # 有一个后继+1个需要用的
                                if self.nodes[s]['type'] == 'out':
                                    droplets.add(0, d.pos.copy(), G.out)
                                    store.ref -= 1
                                elif G.non_dis.in_degree(s) == 0: # 所有前面都完成了 加到CL里
                                    CL.append(s)
                                if store.ref <= 0:
                                    droplets.remove(store)
                                    del op['store']
        for d in ned_remove:
            droplets.remove(d)
        self.check_CL(droplets, chip)



# 将图转换为文本格式
def graph_to_text(G):
    node_lines = []
    arc_lines = []
    sorted_nodes = sorted(G.nodes(data=True), key=lambda x: x[0])
    for node in sorted_nodes:
        node_lines.append(f"{node[0]} {node[1]['htype']} {node[1]['type']} 0")

    for edge in G.edges():
        arc_lines.append(f"ARC a1_0 FROM t1_{edge[0]} TO t1_{edge[1]} TYPE 1")

    return '\n'.join(node_lines + arc_lines)

def protein(splt):
    dim = 2 ** splt
    G = nx.DiGraph()
    m = 6 * dim
    G.add_node(0, htype='ds', type='disS')
    G.add_node(1, htype='ds', type='disB')
    G.add_node(m, htype='conf', type='dlt')
    G.add_edges_from([(1, m), (0, m)])

    k = m + 1
    n = 2

    for i in range(splt):
        for j in range(2 ** (i + 1)):
            G.add_node(k, htype='conf', type='dlt')
            G.add_node(n, htype='ds', type='disB')
            G.add_edges_from([(n, k), (m, k)])
            if j % 2 == 1:
                m += 1
            n += 1
            k += 1

    for i in range(splt, splt + 3):
        for j in range(dim):
            G.add_node(k, htype='conf', type='dlt')
            G.add_node(n, htype='ds', type='disB')
            G.add_edges_from([(n, k), (m, k)])
            m += 1
            n += 1
            k += 1

    i = splt + 3
    for j in range(dim):
        G.add_node(k, htype='conf', type='mix')
        G.add_node(n, htype='ds', type='disR')
        G.add_edges_from([(n, k), (m, k)])
        m += 1
        n += 1
        k += 1
    i = splt + 4
    for j in range(dim):
        G.add_node(k, htype='nonc', type='opt', dur=300)
        G.add_edge(m, k)
        m += 1
        k += 1

    G.porttyp = {'disS': 1, 'disB': 2, 'disR': 2}

    return G


### algorithm-generated assays
def CoDos(y=None):
    G = nx.DiGraph()
    for i in [3,5]:
        G.add_node(i, htype='ds',  type='dis1')
    for i in [0,4]:
        G.add_node(i, htype='ds',  type='dis2')
    for i in [1,7]:
        G.add_node(i, htype='ds', type='dis3')
    for i in [2,6,8]:
        G.add_node(i, htype='ds',  type='dis4')
    for i in range(9,19):
        G.add_node(i, htype='conf', type='dlt')
    G.add_edges_from([(0,9),(1,9),(2,10),(3,11),(4,11),(5,15),(6,15),(7,17),(8,17),
                      (9,10),(9,13),(10,12),(11,12),(11,14),(12,13),(13,14),(14,16),(15,16),(16,18),(17,18)])
    G.porttyp = {'dis1': 2, 'dis2': 2, 'dis3': 2, 'dis4': 3}
    return G

def Gorma(y=None):
    G = nx.DiGraph()
    for i in range(3):
        G.add_node(i, htype='ds', type='dsr')
    for i in range(3, 7):
        G.add_node(i, htype='ds', type='dsb')
    for i in range(7, 15):
        G.add_node(i, htype='conf', type='dlt')
    G.add_edges_from([(0, 7), (1, 8), (2, 9), (3, 7), (4, 12), (5, 13), (6, 14), (7, 8), (7, 10), (8, 9), (8, 11),
                      (9, 10), (10, 11), (11, 12), (12, 13), (13, 14)])
    G.porttyp = {'dsr': 3, 'dsb': 4}
    return G


def Remia(y=None):
    G = nx.DiGraph()
    for i in range(3):
        G.add_node(i, htype='ds', type='dsr')
    for i in range(3, 8):
        G.add_node(i, htype='ds', type='dsb')
    for i in range(8, 17):
        G.add_node(i,htype='conf', type='dlt')
    G.add_edges_from([(0, 8), (3, 8), (4, 9),(5, 10), (6, 11), (7, 12), (8, 14),
                      (9, 13), (1, 15), (2, 16), (8, 9), (9, 10), (11, 12), (13, 14), (15, 16),
                      (10, 11), (12, 13), (14, 15)])
    G.porttyp = {'dsr': 3, 'dsb': 5}
    return G



if __name__ == "__main__":
    splt = 5  # 例如，3层分裂
    G = CoDos()
    graph_text = graph_to_text(G)
    print(graph_text)
    import os
    print(os.getcwd())
    with open("CoDos.txt", "w") as file:
        file.write(graph_text)