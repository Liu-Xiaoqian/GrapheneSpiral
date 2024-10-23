import numpy as np
from numpy.linalg import inv
from numpy import sin,cos,arcsin,arccos,pi,sqrt,ceil
import matplotlib.pyplot as plt

crystal_para = np.array([2.46,2.46,3.35]) #2.467

def make_xyz(xyz,file='data.xyz'):
    xyz_file = open(file,'w')
    print(str(xyz.shape[0])+'\n',file=xyz_file)
    # print('# crystal parameter: ',b[0][1],file=xyz_file)
    for i in range(xyz.shape[0]):
        print('C\t','%.7f\t' % xyz[i][-3], '%.7f\t' % xyz[i][-2], '%.7f\t' % xyz[i][-1], file=xyz_file)
    xyz_file.close()
    return 0

def make_poscar(position,P,file='POSCAR',torrent=0):
    # position = np.delete(position@inv(P),np.where((position@inv(P)>1+torrent)|
    #                                               (position@inv(P)<0-torrent)),axis=0)[0]@P
    print(position.shape[0])
    poscar = open(file,'w')
    print("TBG"+"\n1.000",file=poscar)
    print('\t%.15f' % P[0][0], '\t%.15f' % P[0][1], '\t%.15f' % P[0][2], file=poscar)
    print('\t%.15f' % P[1][0], '\t%.15f' % P[1][1], '\t%.15f' % P[1][2], file=poscar)
    print('\t%.15f' % P[2][0], '\t%.15f' % P[2][1], '\t%.15f' % P[2][2], file=poscar)
    print('C\n','%s\n'%position.shape[0],'C',file=poscar)
    for i in range(position.shape[0]):
        print('\t%.15f'%position[i][0],'\t%.15f'% position[i][1],'\t%.15f'% position[i][2],file=poscar)
    poscar.close()
    return position

def make_lmp(position,P,file='lmp.dat'):
    lmp = open(file,'w')
    print('# LAMMPS data file',file=lmp)
    print('%s'%position.shape[0],'atoms\n'+'1 atom types',file=lmp)
    print('0.0',float(P[0][0]),'xlo xhi',file=lmp)
    print('0.0',float(P[1][1]),'ylo yhi',file=lmp)
    print('0.0',float(P[2][2]),'zlo zhi',file=lmp)
    # print(float(xy),float(xz),float(yz),'xy xz yz\n',file=lmp)
    print(float(P[1][0]),'0.0','0.0','xy xz yz\n',file=lmp)
    print('Masses\n\n 1 12.01\n\nAtoms\n',file=lmp)
    for i in range(position.shape[0]):
        print(int(i+1),'\t1','\t%.7f'%position[i][0],'\t%.7f'%position[i][1],'\t%.7f'%position[i][2],file=lmp)
    lmp.close()
    return 0
    
def unique(xyz):
    dis_mat = np.linalg.norm(xyz[:,None]-xyz[None,:],axis=-1)
    dis_mat = np.triu(dis_mat)
    index = np.argwhere((dis_mat>0)&(dis_mat<1.2))[:,0]
    # index = np.r_[index[:,0],index[:2,1]]
    return np.delete(xyz,index,axis=0)

def test_by_view(position=np.array([0,0,0]),SHOW=True,show_type='plt',show_boundary=False,P=np.diag([0,0,0]),point_size=1,color='b'):
    if show_type == 'plt':
        import matplotlib.pyplot as plt
        plt.scatter(position[:,0],position[:,1],s=point_size,color=color)
        if show_boundary == True:
            va = P[0,:-1]; vb = P[1,:-1]
            v = np.array([[0,0],va,va+vb,vb])
            edges = np.array([
                [v[0],v[1]],
                [v[1],v[2]],
                [v[2],v[3]],
                [v[3],v[0]]
            ])
            for edge in edges:
                plt.plot(edge[:,0],edge[:,1],'black')
        if SHOW == True:
            plt.axis('scaled')
            plt.show()
    elif show_type == 'maya':
        import mayavi.mlab as mlab
        mlab.points3d(position[:,0],position[:,1],position[:,2],scale_factor=point_size)
        if SHOW == True:
            mlab.show()
    return 0

def rotate(original_position,angle,rotate_axis=[0,0,1],dimensions='3D'):
    a,b,c = rotate_axis
    th = np.deg2rad(angle)
    rotation_matrix = np.array([[a*a+(1-a*a)*cos(th),a*b*(1-cos(th))+c*sin(th),a*c*(1-cos(th))-b*sin(th),0],
                                [a*b*(1-cos(th))-c*sin(th),b*b+(1-b*b)*cos(th),b*c*(1-cos(th))+a*sin(th),0],
                                [a*c*(1-cos(th))+b*sin(th),b*c*(1-cos(th))-a*sin(th),c*c+(1-c*c)*cos(th),0],
                                [0,0,0,1]])
    if dimensions == '3D':
        original_position = np.c_[original_position,np.zeros_like(original_position[:,0])].T
        return (rotation_matrix@original_position).T[:,:-1]
    elif dimensions == '2D':
        original_position = np.c_[original_position,np.zeros_like(original_position[:,0]),np.zeros_like(original_position[:,0])].T
        return (rotation_matrix@original_position).T[:,:-2]
    else:
        raise ImportError('input position shape does not fit!')

def build_super_cell(re_mat,P,xyz,coated=False):
    if coated == True:
        m = -int(re_mat[0]/2)
        start = np.array([m,m,0])
    else:
        start = np.array([0,0,0])
    end = start+re_mat
    super_cell = xyz
    for i in range(start[0],end[0]):
        for j in range(start[1],end[1]):
            for k in range(start[2],end[2]):
                re_cell = translate(translate(translate(xyz,P[0]*i),P[1]*j),P[2]*k)
                super_cell = np.append(super_cell,re_cell,axis=0)
    super_cell = super_cell[int(xyz.shape[0]):]
    return super_cell

def translate(original_position,translate_vector):
    x,y,z = translate_vector
    translate_matrix = np.array([[1,0,0,x],
                                 [0,1,0,y],
                                 [0,0,1,z],
                                 [0,0,0,1]])
    original_position = np.c_[original_position, np.ones_like(original_position[:,0])].T
    return (translate_matrix@original_position).T[:,:-1]

def f_theta(m,r=1):
    return (180/pi)*arccos((3*m**2+3*m*r+(r**2)/2)/(3*m**2+3*m*r+r**2))

def hexagonal_grids(n,para=1):
    a = np.linspace(-1,n-1,n+1)
    b = np.linspace(-0.5,n-0.5,n+1)
    A = np.meshgrid(a*sqrt(3),a)
    B = np.meshgrid(b*sqrt(3),b)

    X = np.c_[A[0],B[0]].reshape((n+1)**2*2)-n*sqrt(3)/3
    Y = np.c_[A[1],B[1]].reshape((n+1)**2*2)

    x = np.array([X+1/sqrt(3)*cos(np.deg2rad(60*i)) for i in range(2)]).reshape(4*(n+1)**2)
    y = np.array([Y+1/sqrt(3)*sin(np.deg2rad(60*i)) for i in range(2)]).reshape(4*(n+1)**2)
    return np.array([x,y],dtype=float)*para

def hexagonal_P(crystal=np.array([0,0,0]),hex=np.array([90,90,120]),vacuum=20):
    alpha,beta,gamma = np.deg2rad(hex)
    vector_a = crystal[0]*np.array([1,0,0])
    vector_b = crystal[1]*np.array([cos(gamma),sin(gamma),0])
    # vector_c = crystal[2]*np.array([np.cos(beta),(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma),(1+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma)-np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2)**0.5/np.sin(gamma)])
    vector_c = crystal[2]*np.array([0,0,1])
    P = np.row_stack((vector_a,vector_b,vector_c))
    if vacuum:
        P[2,2] = vacuum
    return P

def make_TBG(m,crystal_para=crystal_para):
    k = 2*m+1
    n = sqrt(m**2+k**2-2*m*k*cos(pi/3))
    P = hexagonal_P(np.array([crystal_para[0]*n,crystal_para[1]*n,crystal_para[2]]),vacuum=30)

    original_grid = hexagonal_grids(int(ceil(P[1,1]/crystal_para[0]/3)*3),crystal_para[0])
    layer_0 = rotate(original_grid.T,f_theta(m)/2,dimensions='2D')
    layer_1 = rotate(original_grid.T,-f_theta(m)/2,dimensions='2D')

    mono_vector = -np.array([crystal_para[0]*sqrt(3)/6,crystal_para[0]/2,0])
    mono_layer = translate(np.c_[original_grid.T,np.zeros_like(original_grid.T[:,0])],mono_vector)
    layer_i = rotate(mono_layer,f_theta(m)/2)[:,:-1]

    twisted = np.r_[np.c_[layer_0,np.ones_like(layer_0[:,0])*crystal_para[2]*1],]
                    # np.c_[layer_i,np.ones_like(layer_i[:,0])*crystal_para[2]*2],]
                    # np.c_[layer_1,np.ones_like(layer_1[:,0])*crystal_para[2]*2]]

    twisted = translate(twisted,[1e-12,1e-12,0])

    TBG = np.delete(twisted@inv(P),np.where((twisted@inv(P)>1)|(twisted@inv(P)<0))[0],axis=0)@P
    return TBG,P
    
def make_graphene(m,crystal_para=crystal_para):
    P = hexagonal_P(np.array([crystal_para[0]*m,crystal_para[1]*m,crystal_para[2]]),vacuum=30)
    original_grid = hexagonal_grids(int(ceil(P[1,1]/crystal_para[0]/3)*3),crystal_para[0])
    layer = translate(np.c_[original_grid.T,np.zeros_like(original_grid.T[:,0])],[1e-12,1e-12,0])
    layer = np.delete(layer@inv(P),np.where((layer@inv(P)>1)|(layer@inv(P)<0))[0],axis=0)@P
    return layer

def read_from_poscar(froot):
    import re
    poscar = open(froot,'r').readlines()
    # system_name = re.sub('\s|\t|\n','',str(poscar[0]))
    # scale_size = np.array(poscar[1],dtype=float)
    lattice_constant = np.loadtxt(poscar[2:5])
    # element_type = re.sub('\t|\n','',str(poscar[5]))
    natom = np.array(poscar[6],dtype=int)
    coordinate = re.sub('\s|\t|\n','',str(poscar[7]))
    if coordinate.startswith('D')|coordinate.startswith('d'):
        position = np.loadtxt(poscar[8:8+natom])@lattice_constant
    else:
        position = np.loadtxt(poscar[8:8+natom])
    return position,lattice_constant

def write_lmp_data(position,vacuum=30,height=7.6,data_file='lmp.dat'):
    atom_num = position.shape[0]
    Position = position[:,1:]
    Position = translate(Position,[-np.min(position[:,-3])+vacuum,
                                   -np.min(position[:,-2])+vacuum,
                                   -np.min(position[:,-1])+vacuum])

    position = np.c_[position[:,0],Position]
    make_poscar(Position,np.eye(3)*[np.max(position[:,-3])+vacuum,
                                    np.max(position[:,-2])+vacuum,
                                    height])
    data_file = open(data_file,'w')
    print('# LAMMPS data file',file=data_file)
    print('%s atoms\n2 atom types'%atom_num,file=data_file)
    print('%.8f\t'%0,'%.8f\t'%(np.max(position[:,-3])+vacuum),'xlo xhi',file=data_file)
    print('%.8f\t'%0,'%.8f\t'%(np.max(position[:,-2])+vacuum),'ylo yhi',file=data_file)
    print('%.8f\t'%0,'%.8f\t'%(height),'zlo zhi',file=data_file)
    print('\n\nAtoms\n',file=data_file)
    for i in range(position.shape[0]):
        print(int(i+1),'\t%.0f'%position[i][0],'\t0','\t%.7f'%position[i][-3],'\t%.7f'%position[i][-2],'\t%.7f'%position[i][-1],file=data_file)
    data_file.close()
    return 0

def make_hex(m=10,edge='zigzag'):
    if edge=='zigzag':
        S,P = make_TBG(m=0)
        S_1 = build_super_cell([m, m, 1], P, S)
    elif edge=='armchair':
        S_1 = make_graphene(m=m)
    S_2 = rotate(S_1, 120)
    S_3 = rotate(S_1, -120)
    S_position = np.r_[S_1, S_2, S_3]
    S_position = unique(S_position)
    return S_position
