import os
import csv
import networkx as nx
import matplotlib.pyplot as plt

def read_matrix_from_csv(filename):
    matrix = []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            matrix.append([int(val) for val in row])
    return matrix

def create_graph_from_adjacency_matrix(matrix, size, filename):
    G = nx.Graph()

    for i in range(size):
        G.add_node(i + 1)

    for i in range(size):
        for j in range(i + 1, size):
            if matrix[i][j] == 1:
                G.add_edge(i + 1, j + 1)
        if matrix[i][i] == 1:
            G.add_edge(i + 1, i + 1)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, labels={i+1: str(i+1) for i in range(size)}, node_size=500, node_color='lightblue', font_size=10)
    plt.title(f"Graph for m{i+1}")

    plt.savefig(filename)
    print(f"Graph saved to {filename}")
    plt.close()

def main():
    # Automatically detect script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_directory = script_dir
    graph_folder = os.path.join(base_directory, 'graphs')

    if not os.path.exists(graph_folder):
        os.makedirs(graph_folder)

    # Use script directory for matrices
    matrices_directory = os.path.join(base_directory, 'exportedMatrices')

    for i in range(2, 12):
        matrix_filename = os.path.join(matrices_directory, f"matrix_m{i}.csv")
        
        if os.path.exists(matrix_filename):
            matrix = read_matrix_from_csv(matrix_filename)
            graph_filename = os.path.join(graph_folder, f"m{i}.png")
            create_graph_from_adjacency_matrix(matrix, len(matrix), graph_filename)
        else:
            print(f"Warning: {matrix_filename} not found, skipping.")

if __name__ == "__main__":
    main()
