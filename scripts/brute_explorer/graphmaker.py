
import networkx as nx


def FindOffset(img):
    for i, item in enumerate(img):
        for j, ite2 in enumerate(item):
            if ite2[0] == 224:
                if ite2[1] == 72:
                    if ite2[2] == 72:
                        return (i,j % 32)


def ImageRecognition(img:str)->tuple[list[list[int]],int]:
    '''
    Melon folly image recoginitioniser.
    '''
    # img=imageio.imread(image)
    # print(img)
    vorm = img.shape
    offset = FindOffset(img)
    pixels = []
    def Recognise(pixel, img, i, j):
        melon = [224, 72, 72, 255]
        lemon = [16, 240, 0, 255]
        flute = [248, 216, 0, 255]
        lowerlemon = [56,152,16,255]
        stick = [80,56,40,255]
        cloud = [248,248,248,255]
        schil = [0,176,0,255]
        limoenschil = [48,200,16, 255]
        if pixel[0] == melon[0]:
            if pixel[1] == melon[1]:
                if pixel[2] == melon[2]:
                    return 'O'
        if pixel[0] == lemon[0]:
            if pixel[1] == lemon[1]:
                if pixel[2] == lemon[2]:
                    return 'L'
        if pixel[0] == flute[0]:
            if pixel[1] == flute[1]:
                if pixel[2] == flute[2]:
                    pixel = img[i+2,j]
                    if pixel[0] == melon[0]:
                        if pixel[1] == melon[1]:
                            if pixel[2] == melon[2]:
                                return 'O'
                    if pixel[0] == lowerlemon[0]:
                        if pixel[1] == lowerlemon[1]:
                            if pixel[2] == lowerlemon[2]:
                                return 'L'
        if pixel[0] == stick[0]:
            if pixel[1] == stick[1]:
                if pixel[2] == stick[2]:
                    pixel = img[i+2,j-1]
                    if pixel[0] == melon[0]:
                        if pixel[1] == melon[1]:
                            if pixel[2] == melon[2]:
                                return 'O'
                    if pixel[0] == stick[0]:
                        if pixel[1] == stick[1]:
                            if pixel[2] == stick[2]:
                                return 'L'
        if pixel[0] == cloud[0]:
            if pixel[1] == cloud[1]:
                if pixel[2] == cloud[2]:
                    pixel = img[i+15,j]
                    if pixel[0] == schil[0]:
                        if pixel[1] == schil[1]:
                            if pixel[2] == schil[2]:
                                return 'O'
                    if pixel[0] == limoenschil[0]:
                        if pixel[1] == limoenschil[1]:
                            if pixel[2] == limoenschil[2]:
                                return 'L'
        return 'W'
    for i in range(offset[0],vorm[0],24):
        row = []
        for j in range(offset[1],vorm[1], 32):
            row.append(Recognise(img[i,j], img, i, j))
        if set(row) != {'W'}:
            pixels.append(row)
    pixels[-1][int((len(pixels[0])-1)/2)] = 'P'
    return pixels

def graphmaker(filename):
    G = nx.Graph()
    matrix = ImageRecognition(filename)
    for i, row in enumerate(matrix):
        for j, item in enumerate(row):
            if item != 'W':
                G.add_node(f"1{i}{j}", tag=item, row=i, col=j)
                if any([node for node in G.nodes(data=True) if ((node[1]['tag'] in {'L','O', 'P'}) and (node[1]['row'] == i) and (node[1]['col'] == j - 1))]):
                    G.add_edge(f"1{i}{j}", f"1{i}{j-1}", color='#cccccc')
                if any([node for node in G.nodes(data=True) if ((node[1]['tag'] in {'L','O', 'P'}) and (node[1]['row'] == i - 1) and (node[1]['col'] == j))]):
                    G.add_edge(f"1{i}{j}", f"1{i -1}{j}", color='#cccccc')  
      
    return G
    


# To visualise the graph
# nx.draw_kamada_kawai(G, node_color=color_map, with_labels = True)
# plt.savefig("test.png")

# To check whether the graph is still fully connected. If this is not the case, the search can be stopped
# print(nx.is_connected(G))