import numpy as np
from supercell import translate, rotate

def cubic_P(xyz,vacuum=15,pbc_z=False):
    xyz = translate(xyz,[-np.min(xyz[:,0])+vacuum,
                         -np.min(xyz[:,1])+vacuum,
                         -np.min(xyz[:,2])+vacuum])
    P = np.diag([np.max(xyz[:,0])-np.min(xyz[:,0])+2*vacuum,
                 np.max(xyz[:,1])-np.min(xyz[:,1])+2*vacuum,
                 np.max(xyz[:,2])-np.min(xyz[:,2])+2*vacuum*int(not pbc_z)])
    return xyz,P
    
def hexagonal_P(crystal=np.array([0,0,0]),hex=np.array([90,90,120]),vacuum_z=20):
    alpha,beta,gamma = np.deg2rad(hex)
    vector_a = crystal[0]*np.array([1,0,0])
    vector_b = crystal[1]*np.array([np.cos(gamma),np.sin(gamma),0])
    #vector_c = crystal[2]*np.array([np.cos(beta),(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma),(1+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma)-np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2)**0.5/np.sin(gamma)])
    vector_c = crystal[2]*np.array([0,0,1])
    P = np.row_stack((vector_a,vector_b,vector_c))
    P[2,2] = vacuum_z if vacuum_z else P[2,2]
    return P

class DataFile():
    def __init__(self,
                 xyz=np.zeros((3,3)),
                 P=np.zeros((3,3)),
                 lattice='hex',
                 pbc=False,
                 f='data'): # **kwargs

        self.xyz = xyz
        self.f = f
        self.natom = self.xyz.shape[0]
        self.lattice = lattice
        self.P = P
        # ScrewDislocation.__init__(self,**kwargs)

        if self.lattice == 'cubic':
            self.xyz,self.P = cubic_P(self.xyz,pbc_z=pbc)
        elif self.lattice == 'hex':
            self.P = self.P

    def write_poscar(self,file_name=False):
        f = file_name if file_name else self.f
        with open(f'{f}.vasp','w') as file:
            file.write(f'{file_name}\n1.0\n')
            for v in self.P:
                file.write('\t'+'\t'.join(['%.15f'%num for num in v])+'\n')
            file.write('C\n %s\n C\n'%self.natom)
            for site in self.xyz:
                file.write('\t'+'\t'.join(map(str, site))+'\n')

    def write_xyz(self,file_name=False):
        f = file_name if file_name else self.f
        with open(f'{f}.xyz','w') as file:
            file.write(f'{self.natom}\n\n')
            for row in self.xyz:
                file.write('C\t'+'\t'.join(map(str, row))+'\n')

    def write_lmp(self,file_name=False,ntype=1,split=1,style='full'):
        f = file_name if file_name else self.f

        atom_id = np.arange(1,self.natom+1)
        mol_id = np.ones(self.natom,dtype=int)
        atom_type = [(i//(self.natom//split))%ntype+1 for i in range(self.natom)]
        q = np.zeros(self.natom,dtype=int)
        x,y,z = self.xyz.T

        style_dict = {
            'angle':[atom_id,mol_id,atom_type,x,y,z],
            'atomic':[atom_id,atom_type,x,y,z],
            'charge':[atom_id,atom_type,q,x,y,z],
            'full':[atom_id,mol_id,atom_type,q,x,y,z],
        }

        with open(f'{f}.lmp','w') as file:
            file.write(f'# LAMMPS, {style} style\n')
            file.write(f'{self.natom} atoms\n')
            file.write(f'{ntype} atom types\n')
            file.write('0\t%s\txlo xhi\n'%self.P[0,0])
            file.write('0\t%s\tylo yhi\n'%self.P[1,1])
            file.write('0\t%s\tzlo zhi\n'%self.P[2,2])
            if self.lattice == 'hex':
                file.write('%s\t0\t0 xy xz yz'%self.P[1,0])
            file.write('\n\nAtoms\n\n')
            for row in zip(*style_dict[style]):
                file.write('\t'.join(map(str, row)) + '\n')

    @staticmethod
    def read_poscar(fpath='POSCAR'):
        import re
        poscar = open(fpath,'r').readlines()
        # system_name = re.sub('\s|\t|\n','',str(poscar[0]))
        # scale_size = np.array(poscar[1],dtype=float)
        lattice = np.loadtxt(poscar[2:5])
        # element_type = re.sub('\t|\n','',str(poscar[5]))
        natom = np.array(poscar[6],dtype=int)
        coordinate = re.sub('\s|\t|\n','',str(poscar[7]))
        if coordinate.startswith('D')|coordinate.startswith('d'):
            xyz = np.loadtxt(poscar[8:8+natom])@lattice
        else:
            xyz = np.loadtxt(poscar[8:8+natom])
        return xyz, lattice

    @staticmethod
    def read_lmp(fpath,outformat='all'):
        import re
        data = open(fpath,'r').readlines()
        lattice = []; xy,xz,yz = np.zeros(3)
        for i,element in enumerate(data):
            atom_num = re.search(r"^(\d+)\satoms",element)
            boundary = re.search(r"^(-?\d.+)[xyz]lo",element)
            rhompara = re.search(r"^(-?\d.+)xy xz yz",element)
            position = re.search(r"^Atoms\s*#\s*(\S+)$",element)

            if atom_num:
                natoms = np.array(atom_num.group(1),dtype=int)

            if boundary:
                lattice.append(boundary.group(1))

            if rhompara:
                xy,xz,yz = np.array(rhompara.group(1).split(),dtype=float)

            if position:
                start = i
                atom_type = position.group(1)

        xlo,xhi,ylo,yhi,zlo,zhi = np.loadtxt(lattice,dtype=float).flatten()
        lattice = np.array([[xhi-xlo,0,0],[xy,yhi-ylo,0],[xz,yz,zhi-zlo]])
        atom_inf = np.loadtxt(data[start+2:start+natoms+2])
        atom_inf = atom_inf[atom_inf[:, 0].argsort()]

        if atom_type == 'full':
            xyz = atom_inf[:,4:7]
        elif atom_type == 'charge':
            xyz = atom_inf[:,3:6]
        elif atom_type == 'atomic':
            xyz = atom_inf[:,2:5]

        if outformat == 'xyz':
            return xyz
        elif outformat == 'all':
            return xyz, lattice
        elif outformat == 'rewrite':
            with open(f'{fpath}.rewrite','w') as file:
                for inf in data[:start+2]:
                    file.write(inf)
                for inf in format_data(atom_inf, atom_inf.shape[1]-7):
                    file.write(' '.join(inf)+'\n')

    @staticmethod
    def read_trj(fpath,select=[0,-1]):
        import re
        trj = open(fpath,'r').readlines()
        assert len(trj)>1
        natom = int(trj[3])
        nframe = len(re.findall(r'ITEM: TIMESTEP', str(trj)))
        select = np.array(select);select[select==-1]=nframe-1
        
#         nframe = int(os.popen(f"grep -o 'TIMESTEP' {fpath} | wc -l").read().strip())
#         natom = int(os.popen(f"grep -A 1 'OF ATOMS' {fpath} | tail -n 1").read().strip())

#         select = np.array(select)
#         select = np.where(select==-1,nframe-1,select)
#         select = np.array([select]) if select.size == 1 else select
        
#         select_range = np.array([select,select+1])*np.array([natom+9])
#         select_line = []
#         for start, end in select_range.T:
#             select_line.append(np.arange(start,end+1))

#         with open(fpath, 'r') as file:
#             for i, line in enumerate(file):
#                 if np.isin(i, np.array(select_line)):
#                     if line.startswith("ITEM: BOX BOUNDS"):
#                         bound.append(line.split())
#                     elif line.startswith("ITEM: ATOMS"):
#                         atoms.append(line.split())
        
        if type(select) == int:
            frame = trj[select*(natom+9):(select+1)*(natom+9)]
        elif type(select) == float:
            frame = [trj[i*(natom+9):(i+1)*(natom+9)] for i in range(int(select*nframe))]

        if read == 'frames':
            select_frames = [trj[i*(natom+9):(i+1)*(natom+9)]
                             for i in select]
        elif (read == 'percent')|(type(select)==float):
            select_frames = [trj[i*(natom+9):(i+1)*(natom+9)]
                             for i in range(int(select[0]*nframe),nframe)]
        elif read == 'single':
            select_frames = [trj[select[0]*(natom+9):(select[0]+1)*(natom+9)]]

        select_frames = np.array(select_frames)
        boundary = np.array([np.loadtxt(select_frames[:,5:8][i]) for i in range(select_frames.shape[0])])
        position = np.array([np.loadtxt(select_frames[:,9:][i]) for i in range(select_frames.shape[0])])
        position = np.array([position[i,np.argsort(position[i][:,0])][:,-3:] for i in range(select_frames.shape[0])])

        if (read == 'percent')|(type(select)==float):
            return position.sum(axis=0)/position.shape[0],boundary.sum(axis=0)/boundary.shape[0]

        return position[0],position[1]

    @staticmethod
    def read_xyz(fpath):
        return np.array(np.loadtxt(fpath,skiprows=2,dtype='str')[:,1:],dtype=float)
        


#xyz,lattice = DataFile.read_lmp('rlx.dat')
