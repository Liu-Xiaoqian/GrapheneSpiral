import numpy as np

def make_xyz(xyz,file='data.xyz'):
    xyz_file = open(file,'w')
    print(str(xyz.shape[0])+'\n',file=xyz_file)
    # print('# crystal parameter: ',b[0][1],file=xyz_file)
    for i in range(xyz.shape[0]):
        print('C\t','%.7f\t' % xyz[i][-3], '%.7f\t' % xyz[i][-2], '%.7f\t' % xyz[i][-1], file=xyz_file)
    xyz_file.close()
    
def make_poscar(position,P,file='POSCAR'):
    poscar = open(file,'w')
    print("TBG"+"\n1.000",file=poscar)
    print('\t%.15f' % P[0][0], '\t%.15f' % P[0][1], '\t%.15f' % P[0][2], file=poscar)
    print('\t%.15f' % P[1][0], '\t%.15f' % P[1][1], '\t%.15f' % P[1][2], file=poscar)
    print('\t%.15f' % P[2][0], '\t%.15f' % P[2][1], '\t%.15f' % P[2][2], file=poscar)
    print('C\n','%s\n'%position.shape[0],'C',file=poscar)
    for i in range(position.shape[0]):
        print('\t%.15f'%position[i][0],'\t%.15f'% position[i][1],'\t%.15f'% position[i][2],file=poscar)
    poscar.close()

def hexagonal_P(crystal=np.array([0,0,0]),hex=np.array([90,90,120]),vacuum=False):
    print('ok')
    alpha,beta,gamma = np.deg2rad(hex)
    vector_a = crystal[0]*np.array([1,0,0])
    vector_b = crystal[1]*np.array([np.cos(gamma),np.sin(gamma),0])
    vector_c = crystal[2]*np.array([np.cos(beta),(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma),(1+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma)-np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2)**0.5/np.sin(gamma)])
    # vector_c = crystal[2]*np.array([0,0,1])
    P = np.row_stack((vector_a,vector_b,vector_c))
    if vacuum:
        P[2,2] = vacuum
    return P

def translate(original_position,translate_vector):
    x,y,z = translate_vector
    translate_matrix = np.array([[1,0,0,x],
                                 [0,1,0,y],
                                 [0,0,1,z],
                                 [0,0,0,1]])
    original_position = np.c_[original_position, np.ones_like(original_position[:,0])].T
    return (translate_matrix@original_position).T[:,:-1]

def rotate(original_position,angle,rotate_axis=[0,0,1],dimensions='3D'):
    from numpy import cos,sin
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

def build_super_cell(re_mat,P,xyz,primitive_cell_cenered=False):
    if primitive_cell_cenered == True:
        start = np.array(re_mat)//2
    else:
        start = np.array([0,0,0])
    end = start+re_mat
    super_cell = xyz
    for k in range(start[2],end[2]):
        for j in range(start[1],end[1]):
            for i in range(start[0],end[0]):
                re_cell = translate(translate(translate(xyz,P[0]*i),P[1]*j),P[2]*k)
                super_cell = np.append(super_cell,re_cell,axis=0)
    super_cell = super_cell[int(xyz.shape[0]):]
    return super_cell, P@np.diag(re_mat)

def get_relative_coor(xyz,P,re=3,dim='2D'):
    supercell = build_super_cell([re,re,1],P,xyz,primitive_cell_cenered=True)
    N1 = int(xyz.shape[0]*np.floor(re**2/2))
    N2 = int(xyz.shape[0]*np.ceil(re**2/2))
    R = (supercell[:, None] - supercell)[N1:N2,N1:N2]
    if dim == '3D':
        return np.triu(R)
    elif dim == '2D':
        return R[np.triu_indices(xyz.shape[0],k=1)]

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
