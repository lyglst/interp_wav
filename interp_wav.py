import numpy as np
import time
import re

def get_psi_from_wannier(w90, k_vecs, supercell, grid):
    '''
    Inverse FFT from wannier function to wavefunction $psi^w$
    to be noticed, $psi^w$ is not eigenfunction of hamiltonian
    Args: 
            w90         :       pywannier90 object
            k_vecs      :       the k-vector list that is being interpolated
            supercell   :       the supercell used for wannier function
            grid        :       the grid density for wannier function
    '''
    grid = np.asarray(grid)
    supercell = np.asarray(supercell)
    k_vecs = np.asarray(k_vecs)
    origin = np.asarray([-(grid[i]*(supercell[i]//2) + 1)/grid[i] for i in range(3)]).dot(w90.real_lattice_loc)       
    real_lattice_loc = (grid*supercell-1)/grid * w90.real_lattice_loc
    nx, ny, nz = grid*supercell
    nband = w90.num_wann

    if w90.spinors:
        wann_up = w90.get_wannier_mod(spinor_mode='up', supercell=supercell, grid=grid).reshape(np.append(grid*supercell, -1))
        wann_down = w90.get_wannier_mod(spinor_mode='down', supercell=supercell, grid=grid).reshape(np.append(grid*supercell, -1))
        #wann_up = w90.get_wannier(spinor_mode='up', supercell=supercell, grid=grid).reshape(np.append(grid*supercell, -1))
        #wann_down = w90.get_wannier(spinor_mode='down', supercell=supercell, grid=grid).reshape(np.append(grid*supercell, -1))
        lpsi_w_up = []
        lpsi_w_down = []
        for kpt in k_vecs:
            x = np.arange(0, supercell[0])
            y = np.arange(0, supercell[1])
            z = np.arange(0, supercell[2])
            xv, yv, zv = np.meshgrid(x, y, z, sparse=False, indexing='ij')
            mesh = np.array(list(zip(xv.flat, yv.flat, zv.flat)))
            mesh_grid = mesh*grid
            #mesh_R = (mesh - supercell//2).dot(w90.real_lattice_loc)
            #phase = np.exp(1.j*kpt.dot(w90.recip_lattice_loc).dot(mesh_R.T))
            phase = np.zeros(grid*supercell, dtype=np.complex128)
            for ix in range((grid*supercell)[0]):
                for iy in range((grid*supercell)[1]):
                    for iz in range((grid*supercell)[2]):
                        phase[ix, iy, iz] = np.exp(
                            1.j*kpt.dot(w90.recip_lattice_loc).dot(
                                (np.array([ix-1, iy-1, iz-1])//grid - supercell//2).dot(w90.real_lattice_loc)))

            psi_w_up = np.zeros(np.append(grid, nband),  dtype=np.dtype('complex128'))
            psi_w_down = np.zeros(np.append(grid, nband),  dtype=np.dtype('complex128'))
            for ix in range(grid[0]):
                for iy in range(grid[1]):
                    for iz in range(grid[2]):
                        for imesh in range(len(mesh_grid)):
                            #psi_w_up[ix, iy, iz] += phase[imesh]*wann_up[ix+mesh_grid[imesh][0]][iy+mesh_grid[imesh][1]][iz+mesh_grid[imesh][2]]
                            #psi_w_down[ix, iy, iz] += phase[imesh]*wann_down[ix+mesh_grid[imesh][0]][iy+mesh_grid[imesh][1]][iz+mesh_grid[imesh][2]]
                            iix = ix + 1 + mesh_grid[imesh][0]
                            iiy = iy + 1 + mesh_grid[imesh][1]
                            iiz = iz + 1 + mesh_grid[imesh][2]
                            if all(np.array([iix, iiy, iiz])<grid*supercell):
                                psi_w_up[ix, iy, iz] += phase[iix, iiy, iiz] * wann_up[iix, iiy, iiz]
                                psi_w_down[ix, iy, iz] += phase[iix, iiy, iiz] * wann_down[iix, iiy, iiz]
            lpsi_w_up.append(psi_w_up)
            lpsi_w_down.append(psi_w_down)
        return lpsi_w_up, lpsi_w_down

def get_psi_ham(w90, k_vecs, psi_w, supercell, grid):
    '''
    transform the wavefunction $psi_w$ to hamiltonian basis
    Args: 
            w90         :       pywannier90 object
            k_vecs      :       the k-vector list that is being interpolated
            psi_w       :       wavefunction $psi_w$
            supercell   :       the supercell used for wannier function
            grid        :       the grid density for wannier function
    '''
    
    if w90.spinors:
        grid = np.asarray(grid)
        supercell = np.asarray(supercell)
        k_vecs = np.asarray(k_vecs)
        psi_w_up, psi_w_down = psi_w
        eig, eigv = w90.interpolate_band(k_vecs, ws_search_size=[4, 4, 4])
        #psi_H_up = np.einsum('pijkm, nmp->pijkn',psi_w_up,eigv.conj().T)
        psi_H_up = np.einsum('pijkm, pmn->pijkn',psi_w_up,eigv)
        #psi_H_down = np.einsum('pijkm, nmp->pijkn',psi_w_down,eigv.conj().T)
        psi_H_down = np.einsum('pijkm, pmn->pijkn',psi_w_down,eigv)
        return psi_H_up, psi_H_down

def get_spin(psi_H_up, psi_H_down):
    spin = []
    for ik in range(len(psi_H_up)):
        spin.append([])
        for iband in range(psi_H_up[ik].shape[3]):
            alpha = np.sum(np.abs(psi_H_up[ik][:,:,:,iband])**2)
            beta = np.sum(np.abs(psi_H_down[ik][:,:,:,iband])**2)
            spin[ik].append((alpha-beta)/(alpha+beta))
    return spin

def is_orth(a, b):
    prod = np.conj(a.flat).dot(np.array(b.flat))
    print('The product is {0}'.format(prod))


def get_s_kpts(w90, op=np.array([[1,0],[0,-1]]), grid=None):
    '''
    get s_k^(H) = <unk|Sz|umk>
    parameter:
        op: the spin operator
        grid: the grid being used for integration
    '''
    if grid is None: grid = w90.wave.ngrid 
    kpts, band, unk = w90.wave.get_wave_nosym(spin=w90.spin, ngrid=grid, norm=False)
    unk_up = [unk[i][:, :grid[0], :, :] for i in range(w90.num_kpts_loc)]
    unk_down = [unk[i][:, grid[0]:, :, :] for i in range(w90.num_kpts_loc)]
    s = np.einsum('sknxyz, st, tkmxyz->knm', np.conj([unk_up, unk_down]), op, np.array([unk_up, unk_down]))
    #np.einsum('sknxyz, st, tknxyz->kn', np.conj([unk_up, unk_down]), np.array([[1,0],[0,1]]), np.array([unk_up, unk_down]))
    norm = np.einsum('sknxyz, st, tknxyz->kn', np.conj([unk_up, unk_down]), np.array([[1,0],[0,1]]), np.array([unk_up, unk_down]))
    norm_mat = np.sqrt([np.outer(norm[i], norm[i]) for i in range(w90.num_kpts_loc)])
    print('Got the S-matrices in low resolution ', time.time())
    return s/norm_mat


def get_s_Rs(w90, Rs, op=2, grid=None):
    '''
    get <0|Sz|R> = \sum_q e^-iqr <u_q^W|Sz|u_q^W>
    where <u_q^W|Sz|u_q^W> = V_q^dagger S_q^(H) V_q
    parameter:
        Rs: the R
    '''
    
    if grid is None: grid = w90.wave.ngrid 
    #s = get_s_kpts(w90, op=op, grid=grid)
    spnx, spny, spnz = get_spn(w90)
    s = np.array([spnx, spny, spnz])[op]
    band_list = np.asarray(w90.band_included_list)
    sw = []
    for k_id in range(w90.num_kpts_loc):
        mo_in_window = w90.lwindow[k_id]
        U_matrix_opt = w90.U_matrix_opt[k_id][:, :np.sum(mo_in_window)].T
        V_matrix = np.einsum('mo,os->ms', U_matrix_opt, w90.U_matrix[k_id].T)
        s_in_window = s[k_id][np.outer(mo_in_window, mo_in_window)].reshape(np.sum(mo_in_window), np.sum(mo_in_window))
        sw.append(np.einsum('ij, jk, km->im', V_matrix.conj().T, s_in_window, V_matrix))
    
    '''
    band_list = np.asarray(w90.band_included_list)
    for k_id in range(w90.num_kpts_loc):
        mo_in_window = w90.lwindow[k_id]
        unk_in_window = unk[k_id][band_list][mo_in_window]
        U_matrix_opt = w90.U_matrix_opt[k_id][:, :np.sum(mo_in_window)].T
        umo_kpt = np.einsum('mxyz,mo,os->sxyz', unk_in_window, U_matrix_opt, w90.U_matrix[k_id].T)
        u_mo_up.append(umo_kpt[:,:grid[0],:,:].reshape(w90.num_wann, -1).T) 
        u_mo_down.append(umo_kpt[:,grid[0]:,:,:].reshape(w90.num_wann, -1).T) 
    sw = np.einsum('skin, st, tkim ->knm', np.conj([u_mo_up, u_mo_down]), op, np.array([u_mo_up, u_mo_down]))
    '''

    # Find the center either R(0,0,0) or the first R in the Rs list
    ngrid = len(Rs)
    nkpts = w90.kpt_latt_loc.shape[0]
    center = np.arange(ngrid)[(np.asarray(Rs)**2).sum(axis=1) < 1e-10]
    if center.shape[0] == 1:
        center = center[0]
    else:
        center = 0
    phase = 1/np.sqrt(nkpts) * np.exp(1j* 2*np.pi * np.dot(Rs, w90.kpt_latt_loc.T))
    sw_r = np.einsum('k,kst,Rk->Rst', phase[center], sw, phase.conj())
    print('Got the S-matrices in real space ', time.time())
    return sw_r
 

def interpolate_s0_kpts(w90, k_vecs, op=2, grid=None, use_ws_distance=True, ws_search_size=[2,2,2], ws_distance_tol=1e-6):
    ndegen, Rs, center = w90.get_wigner_seitz_supercell(ws_search_size, ws_distance_tol)
    if grid is None: grid = w90.wave.ngrid 
    sw_r = get_s_Rs(w90, Rs, op=op, grid=grid)
    
    if use_ws_distance:
        wdist_ndeg, wdist_ndeg_, irdist_ws, crdist_ws = w90.ws_translate_dist(Rs)
        temp = np.einsum('iRstx,kx->iRstk', irdist_ws, k_vecs)
        phase = np.einsum('iRstk,iRst->Rstk', np.exp(1j* 2*np.pi * temp), wdist_ndeg_)
        print('Interpolating the S-matrices ', time.time())
        inter_s0_kpts = \
        np.einsum('R,Rst,Rts,Rstk->kst', 1/ndegen, 1/wdist_ndeg, sw_r, phase) 
        #eig, eigv = w90.interpolate_band(k_vecs, use_ws_distance, ws_search_size, ws_distance_tol)
        print('Interpolating the hamiltonian', time.time())
        eig, eigv = interpolate_ham_from_hr(k_vecs)
        #s0_interp = np.einsum('knm, kmp, kpq->knq',eigv.conj().transpose(0,2,1), inter_s0_kpts, eigv)
        print('Transforming the S-matrices into hamiltonian basis ', time.time())
        s0_interp = np.einsum('knm, kmp, kpq->knq',eigv.transpose(0,2,1), inter_s0_kpts, eigv.conj())
    else:
        phase = np.exp(1j* 2*np.pi * np.dot(Rs, k_vecs.T))
        inter_s0_kpts = \
        np.einsum('R,Rst,Rk->kst', 1/ndegen, sw_r, phase) 
        eig, eigv = w90.interpolate_band(k_vecs, use_ws_distance, ws_search_size, ws_distance_tol)
        s0_interp = np.einsum('knm, kmp, kpq->knq',eigv.conj().transpose(0,2,1), inter_s0_kpts, eigv)
    print('Got the S-matrices in high resolution ', time.time())
    return s0_interp


def get_hr(filename):
    pattern = re.compile(r"[+-]?(?:[0-9]*[.])?[0-9]+")
    f = open(filename, 'r')
    data = f.readline()
    data = f.readline()
    N_band = int(pattern.findall(data)[0])
    data = f.readline()
    ngrid = int(pattern.findall(data)[0])
    ndeg = np.array([])
    for i in range((ngrid-1)//15+1):
        data = f.readline()
        ndeg = np.append(ndeg, np.array(pattern.findall(data), dtype=int))
    hr_mat = np.zeros((ngrid, N_band, N_band), dtype = np.complex128)
    Rs = np.zeros((ngrid,3))
    for i in range(ngrid):
        for n in range(N_band):
            for m in range(N_band):
                data = f.readline()
                numbers = pattern.findall(data)
                hr_mat[i, m, n] = float(numbers[5])+1.j*float(numbers[6])
        Rs[i,0] = int(numbers[0])
        Rs[i,1] = int(numbers[1])
        Rs[i,2] = int(numbers[2])
    return hr_mat, Rs, ndeg

def interpolate_ham_from_hr(k_vecs, filename='wannier90_hr.dat'):  
    hr_mat, Rs, ndegen = get_hr(filename)
    k_vecs = np.array(k_vecs)
    phase = np.exp(1j* 2*np.pi * np.dot(Rs, k_vecs.T))
    inter_hamiltonian_kpts = \
    np.einsum('R,Rst,Rk->kst', 1/ndegen, hr_mat, phase)  
    return np.linalg.eigh(inter_hamiltonian_kpts)     

def get_spn(w90, filename='wannier90.spn'):
    data = np.loadtxt(filename)
    spnx = np.zeros((w90.num_kpts_loc,w90.num_bands_tot,w90.num_bands_tot), dtype = np.complex128)
    spny = np.zeros((w90.num_kpts_loc,w90.num_bands_tot,w90.num_bands_tot), dtype = np.complex128)
    spnz = np.zeros((w90.num_kpts_loc,w90.num_bands_tot,w90.num_bands_tot), dtype = np.complex128) 
    count=0
    for i in range(w90.num_kpts_loc):
        for j in range(w90.num_bands_tot):
            for k in range(j+1):
                spnx[i,k,j]=data[count][0]+1.j*data[count][1]
                spnx[i,j,k]=data[count][0]-1.j*data[count][1]
                count=count+1
                
                spny[i,k,j]=data[count][0]+1.j*data[count][1]
                spny[i,j,k]=data[count][0]-1.j*data[count][1]
                count=count+1
                
                spnz[i,k,j]=data[count][0]+1.j*data[count][1]
                spnz[i,j,k]=data[count][0]-1.j*data[count][1]
                count=count+1
                
    return spnx, spny, spnz
