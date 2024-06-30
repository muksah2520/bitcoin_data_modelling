import csv
import time
import pandas
import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx



def get_num(s):
    s1=""
    s2=""
    ind=-1
    for i in range(len(s)):
        if(s[i]=='E'):
            ind=i
            break
        else:
            s1=s1+s[i]
    if(ind==-1):
        return float(s)
    else:
        for i in range(ind+2,len(s)):
            s2=s2+s[i]
        return float(s1)*pow(10,float(s2))

csv_file_out=open('d27.csv')
csv_file_in=open('d27input.csv')
csv_reader_out=csv.reader(csv_file_out,delimiter=',')
csv_reader_in=csv.reader(csv_file_in,delimiter=',')
line_count=0
dict_out={}
dict_in={}

for row in csv_reader_out:
    if(row[9]=='0'):
        thash=row[1]
        if(thash in dict_out):
            l=dict_out[thash]
            l.append([get_num(row[4])/(100000000),row[6]])
            dict_out[thash]=l
        else:
            dict_out[thash]=[[get_num(row[4])/(100000000),row[6]]]
for row in csv_reader_in:
    if(row[4]=="value"):
        continue
    thash=row[12]
    if(thash in dict_in):
        l=dict_in[thash]
        l.append([(get_num(row[4])/100000000) , row[6] ])
        dict_in[thash]=l
    else:
        dict_in[thash]=[[ (get_num(row[4])/100000000) , row[6] ]]
node_ind=0
dict_nodes={}
dict_edges={}
dict_nodes2={}
for key in dict_out:
    lo=dict_out[key]
    li=dict_in[key]
    lo.sort(reverse=True)
    li.sort(reverse=True)
    i=j=0
    while(len(lo)>0 and len(li)>0):
        lo.sort(reverse=True)
        li.sort(reverse=True)
        r=lo[i][1]
        s=li[j][1]
        if(s not in dict_nodes):
            dict_nodes[s]=node_ind
            dict_nodes2[node_ind]=s
            node_ind=node_ind+1
        if(r not in dict_nodes):
            dict_nodes[r]=node_ind
            dict_nodes2[node_ind]=r
            node_ind=node_ind+1

        if(lo[i][0]==li[j][0]):
            if((dict_nodes[s],dict_nodes[r]) not in dict_edges):
                dict_edges[(dict_nodes[s],dict_nodes[r])]=[li[j][0],1]
            else:
                l=dict_edges[(dict_nodes[s],dict_nodes[r])]
                l[0]+=li[j][0]
                l[1]=l[1]+1
                dict_edges[(dict_nodes[s],dict_nodes[r])]=l
            lo.pop(0)
            li.pop(0)
            #i=i+1
            #j=j+1
        elif(lo[i]<li[j]):
            
            if((dict_nodes[s],dict_nodes[r]) not in dict_edges):
                dict_edges[(dict_nodes[s],dict_nodes[r])]=[lo[i][0],1]
            else:
                l=dict_edges[(dict_nodes[s],dict_nodes[r])]
                l[0]+=lo[i][0]
                l[1]=l[1]+1
                dict_edges[(dict_nodes[s],dict_nodes[r])]=l
            li[j][0]-=lo[i][0]
            lo.pop(0)
            #i=i+1
        else:
            if((dict_nodes[s],dict_nodes[r]) not in dict_edges):
                dict_edges[(dict_nodes[s],dict_nodes[r])]=[li[j][0],1]
            else:
                l=dict_edges[(dict_nodes[s],dict_nodes[r])]
                l[0]+=li[j][0]
                l[1]=l[1]+1
                dict_edges[(dict_nodes[s],dict_nodes[r])]=l
            lo[i][0]-=li[j][0]
            li.pop(0)
            #j=j+1

#graph formation         
le1=[]
le2=[]
l_edg_attr=[]
how=20
curr=0
for edges in dict_edges.keys():
    curr=curr+1
    if(curr==how):
        break
    le1.append(edges[0])
    le2.append(edges[1])
    l_edg_attr.append([dict_edges[edges][0],dict_edges[edges][1]])

l_x=[]
node_labels={}
for i in range(how):
    #node_labels[i]=dict_nodes2[i] #(for public key as node labels)
    node_labels[i]=i
    l_x.append([i])

x=torch.tensor(l_x,dtype=torch.float)
edge_index=torch.tensor([le1,le2],dtype=torch.long)

data=Data(x=x,edge_index=edge_index)

# Convert the PyTorch Geometric data to NetworkX graph
G = to_networkx(data, node_attrs=['x'])
for node, label in node_labels.items():
    G.nodes[node]['label'] = label

# Plot the graph using NetworkX and Matplotlib
plt.figure(figsize=(200, 200))

pos = nx.spring_layout(G)  # Position nodes using Fruchterman-Reingold force-directed algorithm

# Draw nodes with their attributes
nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_size=200, node_color='skyblue', font_size=16, font_color='black')
# Draw edge labels
edge_labels = {(i, j): f'{dict_edges[(i,j)][0]}-{dict_edges[(i,j)][1]}' for i, j in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')


#manual checking
print("node ind: ",dict_nodes["17zQZ3zHaBDv2Lo95dWHkgFdfDPBxLruQ6"])
print("node ind: ",dict_nodes["14Bf8QywmcCHurft4xgwgPdRoBWbG9ptjM"])
print("node ind: ",dict_nodes["17GaCE8pJcVrvX9GAhWDEXwMgxURhwwHV7"])
print("node ind: ",dict_nodes["1Bp4wjqapUDaPtjAgPXEQedb1BWUEs1m5w"])


print("node ind: ",dict_nodes["1QL314FBUyCPSer7MzYHy3ALs2GswqXKPE"])
print("node ind: ",dict_nodes["1EWaHfec7Hx6FfNgvs4kPqv45v9pMGk5i8"])
print("node ind: ",dict_nodes["183HzvAogpGpgvurkf8Af7ZMPXyxECuve5"])
print("node ind: ",dict_nodes["1LzmPUBJgp8t6tE6TrWkfRL8zN1NoN1uKU"])

#manual checking2
print("node ind: ",dict_nodes["1Ky3FvWTY6i9uLKnggzAJq3aBUvNMGDMuH"])
print("node ind: ",dict_nodes["1PWVLc4A1U953UxpNrUJ9y8VpDW4TpSuf6"])
print("node ind: ",dict_nodes["1EtksATHtr6q3Va2qhEvoWoUgJknnDAVA6"])
print("node ind: ",dict_nodes["1ARYzq8bkZaetukFvtmDXu2c4f4muBkCpP"])
print("node ind: ",dict_nodes["1HdxdoHkrSCFmSCzMhA2rUD9hgyYoLHMSN"])

print("gg")

print(dict_nodes2[43])
print(dict_nodes2[28])
print(dict_nodes2[19])
print(dict_nodes2[38])
print(dict_nodes2[97])
print(dict_nodes2[12])
plt.show()

