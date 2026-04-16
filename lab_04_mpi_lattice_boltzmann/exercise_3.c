/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

//////////////////////////////////////////////////////
//
// Goal: Implement non-blocking 1D communication scheme
//       along X axis.
//
// SUMMARY:
//     - 1D splitting along X
// NEW:
//     - >>> Non-blocking communications <<<
//
//////////////////////////////////////////////////////

/****************************************************/
#include "src/lbm_struct.h"
#include "src/exercises.h"

/****************************************************/
void lbm_comm_init_ex3(lbm_comm_t * comm, int total_width, int total_height)
{
	//we use the same implementation then ex1
	lbm_comm_init_ex1(comm, total_width, total_height);
}

/****************************************************/
void lbm_do_step_ex3(lbm_comm_t * comm, lbm_mesh_type_t * mesh_type, lbm_mesh_t * mesh, lbm_mesh_t * temp_mesh)
{
    lbm_phys_special_cells(mesh, mesh_type, comm);

    
    lbm_phys_collision(temp_mesh, mesh);

    
    lbm_comm_ghost_exchange_ex3(comm, temp_mesh);  

    //overlapping computation on the inner cell(non depending on ghost cells)
    lbm_phys_propagation_inner(mesh, temp_mesh);

    
    lbm_comm_ghost_exchange_ex3_wait(comm);

    
    lbm_phys_propagation_border(mesh, temp_mesh);
}
void lbm_comm_ghost_exchange_ex3(lbm_comm_t * comm, lbm_mesh_t * mesh)
{
	
	int rank, comm_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	int count = comm->height * DIRECTIONS;
	comm->req_count=0;

	double * real_cell_first=lbm_mesh_get_cell(mesh,1,0);
	double * real_cell_last=lbm_mesh_get_cell(mesh,comm->width-2,0);

	double * ghost_cell_left=lbm_mesh_get_cell(mesh,0,0);
	double * ghost_cell_right=lbm_mesh_get_cell(mesh,comm->width-1,0);

	//recv from the left into ghost cell
	if (rank>0){
		MPI_Irecv(ghost_cell_left,count,MPI_DOUBLE,rank-1,0,comm->communicator,&comm->requests[comm->req_count++]);
	}
	//recv from the right into ghost cell
	if(rank<comm_size-1){
		MPI_Irecv(ghost_cell_right,count,MPI_DOUBLE,rank+1,0,comm->communicator,&comm->requests[comm->req_count++]);
	}

	//send first real col to rank-1

	if(rank>0){
		MPI_Isend(real_cell_first,count,MPI_DOUBLE,rank-1,0,comm->communicator,&comm->requests[comm->req_count++]);
	}
	//send last real col to rank+1
	if(rank<comm_size-1){
		MPI_Isend(real_cell_last,count,MPI_DOUBLE,rank+1,0,comm->communicator,&comm->requests[comm->req_count++]);
	}


}

void lbm_comm_ghost_exchange_ex3_wait(lbm_comm_t * comm)
{
    if (comm->req_count > 0)
        MPI_Waitall(comm->req_count, comm->requests, MPI_STATUS_IGNORE);
}
