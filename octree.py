import numpy as np

class node:
    def __init__(self, box, particles, masses):
        """
        box: a bbox object with the dimensions of the node/cell
        particles: a numpy array of shape(x,3) of particle positions that are under this node/leaf's subtree
        masses: a numpy array of shape(x,1) of particle masses that are under this node/leaf's subtree
        """
        self.box = box
        self.center = self.box.middle() #center of node/cell
        self.children = [] #list of children of this node
        
        self.n = 0 #number of particles 
        self.p = None #the *single* particle bound to this node. only set when we hit a leaf node
        
        self.particles = particles #positions of particles belonging to this node
        self.masses = masses #masses of particles belonging to this node
        
        self.M = np.sum(self.masses) #total mass of all particles in this node
        
    def insert(self, particle):
        """
        particle: (3,) numpy array of the position of a particle to be added to the current node
        
        This is a recursive function that will keep creating new nodes/cells/octants till it finds a node
        with 0 particles in it and then populate that node with the input particle.
        """
        if(not self.box.inside(particle)): #check if particle is in the node/cell
            return 
        
        if(self.n == 0): #no particles in this node/cell
            self.p = particle #assign particle to this node/cell
            self.com = particle #com is just the particle position 
        else:
            if(self.n == 1): #leaf node
                self.create_children(self.box) #create children (aka subtree)
                for child in self.children:
                    child.insert(self.p) #need to reassign the current particle bc the new particle also wants to be in this node/cell
                self.p = None #once it's reassigned, the particle belonging to this node/cell becomes none
            for child in self.children:
                child.insert(particle) #iterate over the children and assign the input particle
                
        self.update_com() #update center of mass of this node/cell once the particle is assigned
        self.n += 1 #update number of particles belonging to this node/cell.
    
        return 
    
    def update_com(self):
        """
        updates center of mass. p self-explanatory
        """
        self.com = np.zeros(3)
        self.com[0] = np.dot(self.particles[:,0], self.masses[:])/self.M
        self.com[1] = np.dot(self.particles[:,1], self.masses[:])/self.M
        self.com[2] = np.dot(self.particles[:,2], self.masses[:])/self.M
        
        return 
    
    def create_children(self, box):
        """
        subdivides the current box into 8 octants and creates those children nodes. 
        """
        xhalf = self.center[0]
        yhalf = self.center[1]
        zhalf = self.center[2]
        
        index = self.particles > self.box.middle() #will return a boolean array of shape (len(particles),3) indicating which octant the particles belong in
        
        c1_box = bbox(np.array([[box.xlow, xhalf], [box.ylow, yhalf], [box.zlow, zhalf]]))
        mask = np.all(index == np.bool_([0,0,0]), axis=1)
        c1 = node(c1_box, self.particles[mask], self.masses[mask])
        
        c2_box = bbox(np.array([[xhalf, box.xhigh], [box.ylow, yhalf], [box.zlow, zhalf]]))
        mask = np.all(index == np.bool_([1,0,0]), axis=1)
        c2 = node(c2_box, self.particles[mask], self.masses[mask])
        
        c3_box = bbox(np.array([[box.xlow, xhalf], [yhalf, box.yhigh], [box.zlow, zhalf]]))
        mask = np.all(index == np.bool_([0,1,0]), axis=1)
        c3 = node(c3_box, self.particles[mask], self.masses[mask])
        
        c4_box = bbox(np.array([[xhalf, box.xhigh], [yhalf, box.yhigh], [box.zlow, zhalf]]))
        mask = np.all(index == np.bool_([1,1,0]), axis=1)
        c4 = node(c4_box, self.particles[mask], self.masses[mask])
        
        c5_box = bbox(np.array([[box.xlow, xhalf], [box.ylow, yhalf], [zhalf, box.zhigh]]))
        mask = np.all(index == np.bool_([0,0,1]), axis=1)
        c5 = node(c5_box, self.particles[mask], self.masses[mask])
        
        c6_box = bbox(np.array([[xhalf, box.xhigh], [box.ylow, yhalf], [zhalf, box.zhigh]]))
        mask = np.all(index == np.bool_([1,0,1]), axis=1)
        c6 = node(c6_box, self.particles[mask], self.masses[mask])
        
        c7_box = bbox(np.array([[box.xlow, xhalf], [yhalf, box.yhigh], [zhalf, box.zhigh]]))
        mask = np.all(index == np.bool_([0,1,1]), axis=1)
        c7 = node(c7_box, self.particles[mask], self.masses[mask])
        
        c8_box = bbox(np.array([[xhalf, box.xhigh], [yhalf, box.yhigh], [zhalf, box.zhigh]]))
        mask = np.all(index == np.bool_([1,1,1]), axis=1)
        c8 = node(c8_box, self.particles[mask], self.masses[mask])
            
        self.children = [c1, c2, c3, c4, c5, c6, c7, c8] #assign children

class bbox:
    def __init__(self, box, dim=3):
        """
        bbox makes life a lil easier. 
        box: numpy array (3,2)
        """
        self.bb = np.array(box)
        
        self.xlow = self.bb[0,0]
        self.xhigh = self.bb[0,1]
        
        self.ylow = self.bb[1,0]
        self.yhigh = self.bb[1,1]
        
        self.zlow = self.bb[2,0]
        self.zhigh = self.bb[2,1]
        
        self.center = np.array([(self.bb[0,0]+self.bb[0,1])/2, (self.bb[1,0]+self.bb[1,1])/2, (self.bb[2,0]+self.bb[2,1])/2])
        self.dim = dim
     
    def __call__(self):
        return self.bb
    
    def inside(self, p):
        """
        given coordinate is inside the bounding box
        input: p is an array of x, y, z
        output: True or False
        """
        if (p[0] < self.xlow or p[0] > self.xhigh or p[1] < self.ylow or p[1] > self.yhigh or p[2] < self.zlow or p[2] > self.zhigh):
            return False
        else:
            return True
        
    def middle(self):
        """
        returns center of bounding box
        """
        return np.array([((self.xlow + self.xhigh))/2., (self.ylow + self.yhigh)/2., (self.zlow + self.zhigh)/2.])



