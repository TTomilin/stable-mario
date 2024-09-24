import graphmaker
import networkx as nx
import matplotlib.pyplot as plt

def evaluation(graph, current, end):
    score = 0
    neighbors = list(graph.neighbors(current))
    graph.remove_node(current)
    if not nx.is_connected(graph):
        return score
    for neighbor in neighbors:
        if neighbor == end or neighbor == current:
            continue
        elif len(list(graph.neighbors(neighbor))) == 1:
            score += 1
            if score == 2:
                return 0
    return (int(current[1])-int(end[1]))**2+(int(current[2])-int(end[2]))**2

def main(img):
    G = graphmaker.graphmaker(img)
    currentnode = [x for x,y in G.nodes(data=True) if y['tag'] == 'P'][0]
    endnote = [x for x,y in G.nodes(data=True) if y['tag'] == 'L'][0]
    vistednodes = [currentnode]
    numberofnodes = G.number_of_nodes()
 
    def RBFS(graph, visited, current, end):
        
        if len(visited) == numberofnodes and visited[-1]==endnote:
            return visited
        neighbors = list(graph.neighbors(current))
        H = graph.copy()
        if len(neighbors) == 1:
            H.remove_node(current)
            J = H.copy()
            visited.append(neighbors[0])
            J = RBFS(J, visited, neighbors[0], end)
            if len(visited) == numberofnodes and visited[-1]==endnote:
                return visited
            visited.pop()
        else:
            scores = []
            for i, neighbor in enumerate(neighbors):
                P = H.copy()
                P.remove_node(current)
                score = evaluation(P, neighbor, end)
                if score == 0:
                    neighbors[i] = "None"
                scores.append(score)
            neighbors = [x for x in [x for _,x in sorted(zip(scores, neighbors), reverse=True)] if x != "None"]
            H.remove_node(current)
            for neighbor in neighbors:
                J = H.copy()
                visited.append(neighbor)
                J = RBFS(J, visited, neighbor, end)
                if len(visited) == numberofnodes and visited[-1]==endnote:
                    return visited
                visited.pop() 
    result = RBFS(G, vistednodes, currentnode, endnote) 
    # #to graph
    # colors = {
    #     "L": "green",
    #     "O": "red",
    #     "P": "blue"
    # }
    
    # color_map = [None]*200
    # for node in result:
    #     color_map[int(node)-100] = colors[G.nodes[node]["tag"]]
    # G = nx.relabel_nodes(G, dict(zip(result, range(len(result)))), copy=False)
    # for i in range(numberofnodes):
    #     if i != 0:
    #         G[i-1][i]['color'] = '#f80000'
    # color_map = [x for x in color_map if x != None]
    # edges = G.edges()
    # edge_colors = [G[u][v]['color'] for u,v in edges]
    # nx.draw_kamada_kawai(G, node_color=color_map, edge_color=edge_colors, with_labels = True)
    # plt.savefig("test23.png") 
    return result
# print(main())