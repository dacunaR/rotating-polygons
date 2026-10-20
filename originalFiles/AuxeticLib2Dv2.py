import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import collections
import pickle
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize
from matplotlib import cm
import copy

def Rot(theta):
    return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

class Auxetic(collections.abc.Sequence):
    def __init__(self,filename):


        self.lam=1
        self.Lx=1
        self.Ly=1
        self.data=[]
        self.makeNet(filename)
        self.fillPoints()

        
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)
    
    def makeNet(self,fileName):
        file=open(fileName)
        flag=0
        index=0
        data=[]
        for line in file:
            if flag==1: #just to skip the first line
                arLine=[float(i) for i in line.split() if i!='-1']
                data.append(Poli(index,arLine[0],arLine[1],[int(i) for i in arLine[2:]] ))
                index+=1
            else: #set Lx and Ly
                arLine=[float(i) for i in line.split() if i!='-1']
                self.Lx=arLine[0]
                self.Ly=arLine[1]
            flag=1
        file.close()
        self.data=data

    def fillPoints(self):
        for poli in self.data:
            for index in poli.vecinosId:
                poli.patas.append(Pata(poli,self[index]))
        for poli in self.data:
            for pata in poli.patas:
                if pata.pataVecina==0:
                    neigh=pata.myVecino
                    for pataN in neigh.patas:
                        if pataN.myVecino==poli:
                            pata.pataVecina=pataN
                            pataN.pataVecina=pata
                            break
        for poli in self.data:
            for pata in poli.patas:
                pataVecina=pata.pataVecina
                neigh=pataVecina.myPoli
                dx,dy=calcDist(poli,neigh,self.Lx,self.Ly)
                pata.a=np.linalg.norm([dx,dy])/2.0
                pata.alpha=np.arctan2(-dy,-dx)
                pata.avec0=np.array([np.cos(pata.alpha),np.sin(pata.alpha)])*pata.a
            poli.setPos(poli.x,poli.y,poli.theta)
            poli.patas=sorted(poli.patas, key=lambda x: x.alpha)

    def getVar(self):
        vec=[]
        for poli in self:
            vec.append(poli.x)
            vec.append(poli.y)
            vec.append(poli.theta)
        return np.array(vec)
    def setVar(self,vec):
        i=0
        for poli in self:
                        #x      #y          #theta
            poli.setPos(vec[3*i],vec[3*i+1],vec[3*i+2])
            i+=1

    def bipartiteAssign(self):
        for poli in self:
            poli.bipId=0
        BipAssing(self[0].vecinos[0]) #red cuadrada uso self[0]

    
    def plot(self):
        plt.figure()
        for poli in self:
            avecs=[np.array(pata.avec)+np.array([poli.x,poli.y]) for pata in poli.patas]
            if len(avecs)<3:
                avecs.append([poli.x,poli.y])
            avecs=np.array(avecs)
            plt.gca().add_patch(matplotlib.patches.Polygon(avecs,edgecolor='black'))
        plt.axis('scaled')
        #plt.axis('off')
        #plt.show()
        
    def plotColor(self):
        #plt.figure()
        fig, ax = plt.subplots()
        colorMap= cm.get_cmap('cividis', 10000)
        polygons=[]
        colors=[]
        for poli in self:
            colors.append(colorMap(np.sin(poli.theta*poli.bipId)/2+1/2))
            polygons.append(matplotlib.patches.Polygon(poli.avec+[poli.x,poli.y]))#,color=colorMap(np.sin(poli.theta*poli.bipId)/2+1/2)))
            #plt.gca().add_patch(matplotlib.patches.Polygon(poli.avec+[poli.x,poli.y],color=colorMap(np.sin(poli.theta*poli.bipId)/2+1/2)))#,color=poli.theta*poli.bipId))
        p=matplotlib.collections.PatchCollection(polygons)
        p.set_color(np.array(colors))
        ax.add_collection(p)
        plt.axis('scaled')
    
    def plotFig(self,image):
        #plt.figure()
        Ar=self
        colorMap= cm.get_cmap('tab20', 10000)

        fig, ax = plt.subplots()

        if image=="poli" or image=="both":
            polygons=[]
            colors=[]
            edgecolor=[]
            for poli in Ar:
                #colors.append(colorMap(np.sin(poli.theta)+1/2))
                if poli.bipId==1:
                    colors.append(colorMap(1/20))
                    edgecolor.append(colorMap(0/20))
                else:
                    colors.append(colorMap(11/20))
                    edgecolor.append(colorMap(10/20))
                polygons.append(matplotlib.patches.Polygon(poli.avec+[poli.x,poli.y]))#,color=colorMap(np.sin(poli.theta*poli.bipId)/2+1/2)))
                #plt.gca().add_patch(matplotlib.patches.Polygon(poli.avec+[poli.x,poli.y],color=colorMap(np.sin(poli.theta*poli.bipId)/2+1/2)))#,color=poli.theta*poli.bipId))
            p=matplotlib.collections.PatchCollection(polygons)
            p.set_color(np.array(colors))
            p.set_edgecolor(np.array(edgecolor))
            p.set_linewidth(2.5)
            ax.add_collection(p)
            plt.axis('scaled')
        if image=="net" or image=="both":
            for point in Ar:
                for neigh in point.vecinos:
                    if neigh.id<point.id:
                        neighx=neigh.x
                        neighy=neigh.y
                        pointx=point.x
                        pointy=point.y
                        flag=0
                        if np.abs(point.x-neigh.x)>Ar.Lx/2:
                            flag=1
                            if point.x>neigh.x:
                                neighx=neighx+Ar.Lx
                                pointx=pointx-Ar.Lx
                            else:
                                neighx=neighx-Ar.Lx
                                pointx=pointx+Ar.Lx
                        if np.abs(point.y-neigh.y)>Ar.Ly/2:
                            flag=1
                            if point.y>neigh.y:
                                neighy=neighy+Ar.Ly
                                pointy=pointy-Ar.Ly
                            else:
                                neighy=neighy-Ar.Ly
                                pointy=pointy+Ar.Ly
                        if flag==0:
                            plt.plot([point.x,neighx],[point.y,neighy],'-',color='black')
                        else:
                            plt.plot([point.x,neighx],[point.y,neighy],'-',color='black')
                            plt.plot([pointx,neigh.x],[pointy,neigh.y],'-',color='black')

            for point in Ar.data:
                if point.bipId==1:
                    myEcolor=colorMap(0/20)
                    myFcolor=colorMap(1/20)
                else:
                    myEcolor=colorMap(10/20)   
                    myFcolor=colorMap(11/20)    
                plt.plot(point.x,point.y,'o',markersize='6',mec=myEcolor,mfc=myFcolor,mew=2)


        plt.axis('scaled')
        margin=0.00
        plt.xlim(0-margin,Ar.Lx+margin)
        plt.ylim(0-margin,Ar.Ly+margin)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        
    def save(self,name=''):
        pickle.dump(self, open(name+'auxetic.p', 'wb'))

##############################FIN CLASE AUXETIC###################################################  
        
        
class Poli:
    def __init__(self,id,x,y,vecinosId):
        self.id=id
        self.theta=0.0#(np.random.rand()-0.5)*2*np.pi/1000
        self.alphas=[]
        self.x=x
        self.y=y
        self.vecinos=[]
        self.vecinosId=vecinosId
        self.a=[] #tamaño de cada lado
        self.avec0=[]
        self.avec=[]
        self.avecp=[]
        self.static=0
        self.patas=[]
        
        
    def setPos(self,x,y,theta):
        self.theta=theta
        self.x=x
        self.y=y
        self.updatePoli()
            
    def updatePoli(self):
        self.R=Rot(self.theta)
        for pata in self.patas:
            pata.updatePata()

class Pata:
    def __init__(self,myPoli,myVecino):
        self.myPoli=myPoli
        self.myVecino=myVecino
        self.pataVecina=0
        self.alpha=0   #angulo de rotacion en el plano xy
        self.a=0
        self.avec=[]
        self.avec0=[]  #vector a, considerando que aun no hay rotacion
        self.valid=1
        self.id=[]

    def updatePata(self): #update de el vector a
        self.avec=self.myPoli.R.dot(self.avec0)

##############################FIN CLASE POLIGONOS###################################################        
def calcDist(poli1,poli2,Lx,Ly):
    distx=poli1.x-poli2.x
    disty=poli1.y-poli2.y
    if np.abs(disty)>Ly/2:
        if disty>0:
            disty=disty-Ly
        else:
            disty=Ly+disty    
    if np.abs(distx)>Lx/2:
        if distx>0:
            distx=distx-Lx
        else:
            distx=Lx+distx
    return np.array([distx,disty])

def BipAssing(poli):
    poli.bipId=1
    Q=[]
    Q.append(poli)
    while(Q!=[]):
        v=Q[0]
        Q.remove(v)
        if v.bipId==1:
            newId=-1
        else:
            newId=1
        for neigh in v.vecinos:
            if neigh.bipId==0:
                neigh.bipId=newId
                Q.append(neigh)
            elif neigh.bipId==v.bipId:
                print("Error Not Bipartite")
                return -1


def recBipAssing(myid,poli):
    poli.bipId=myid
    if myid==1:
        newid=-1
    else:
        newid=1
    for neigh in poli.vecinos:
        if neigh.bipId==0:
            recBipAssing(newid,neigh)
        elif neigh.bipId==myid:
            print("Error Not Bipartite")
            return -1
    
# def cutLink(point1,point2):
#     for pata in point1.patas:
#         if point2==pata.myVecino:
#             point1.patas.remove(pata)
#             pata1=pata
#             break
#     for pata in point2.patas:
#         if point1==pata.myVecino:
#             pata2=pata
#             break
#     point2.patas.remove(pata2)
    
    
def cutLink(point1,point2):
    for pata in point1.patas:
        if point2==pata.myVecino:
            point1.patas.remove(pata)
            break
    for pata in point2.patas:
        if point1==pata.myVecino:
            point2.patas.remove(pata)
            break



def removeNode(point,net):
    copypatas=copy.copy(point.patas)
    for pata in copypatas:
        neigh=pata.myVecino
        cutLink(neigh,point)
    for poli in net:
        if poli.id>point.id:
            poli.id=poli.id-1 
    net.data.remove(point)

def removeLoneNodes(Ar):
    flag=0
    while flag==0:
        flag=1
        for poli in Ar:
            if len(poli.patas)<2:
                removeNode(poli,Ar)
                flag=0

def cutLinkV(point1,point2):
    for pata in point1.patas:
        if point2==pata.myVecino:
            pata.valid=0
            break
    for pata in point2.patas:
        if point1==pata.myVecino:
            pata.valid=0
            break

def removeNodeV(point,net):
    copypatas=copy.copy(point.patas)
    for pata in copypatas:
        neigh=pata.myVecino
        cutLinkV(neigh,point)
    for poli in net:
        if poli.id>point.id:
            poli.id=poli.id-1 
    net.data.remove(point)

def removeLoneNodesV(Ar):
    flag=0
    while flag==0:
        flag=1
        for poli in Ar:
            c=0
            for pata in poli.patas:
                c+=pata.valid
            if c<2:
                removeNode(poli,Ar)
                flag=0