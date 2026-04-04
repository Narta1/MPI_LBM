/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

//////////////////////////////////////////////////////
//
// Goal: Implement odd/even 1D blocking communication scheme 
//       along X axis.
//
// SUMMARY:
//     - 1D splitting along X
//     - Blocking communications
// NEW:
//     - >>> Odd/even communication ordering <<<<
//
//////////////////////////////////////////////////////

/****************************************************/
#include "src/lbm_struct.h"
#include "src/exercises.h"

/****************************************************/
void lbm_comm_init_ex2(lbm_comm_t * comm, int total_width, int total_height)
{
	//we use the same implementation then ex1
	lbm_comm_init_ex1(comm, total_width, total_height);
}

/****************************************************/
void lbm_comm_ghost_exchange_ex2(lbm_comm_t * comm, lbm_mesh_t * mesh)
{
	//
	// TODO: Implement the 1D communication with blocking MPI functions using
	//       odd/even communications.
	//
	// To be used:
	//    - DIRECTIONS: the number of doubles composing a cell
	//    - double[9] lbm_mesh_get_cell(mesh, x, y): function to get the address of a particular cell.
	//    - comm->width : The with of the local sub-domain (containing the ghost cells)
	//    - comm->height : The height of the local sub-domain (containing the ghost cells)
	
	//example to access cell
	//double * cell = lbm_mesh_get_cell(mesh, local_x, local_y);
	//double * cell = lbm_mesh_get_cell(mesh, comm->width - 1, 0);

	int rank=MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	int comm_size=MPI_Comm_size(MPI_COMM_WORLD,&comm_size);

	int count = comm->height * DIRECTIONS;

	double * real_cell_first=lbm_mesh_get_cell(mesh,1,0);
	double * real_cell_last=lbm_mesh_get_cell(mesh,comm_size-2,0);

	double * ghost_cell_left=lbm_mesh_get_cell(mesh,0,0);
	double * ghost_cell_right=lbm_mesh_get_cell(mesh,comm_size-1,0);

	//communication to the left :
	if (rank>0) {
		if(rank%2!=0){
			MPI_Send(real_cell_first,count,MPI_DOUBLE,rank-1,0,comm->communicator);
			MPI_Recv(ghost_cell_left,count,MPI_DOUBLE,rank-1,0,comm->communicator,MPI_STATUS_IGNORE);
		} 
		else{
			MPI_Recv(ghost_cell_left,count,MPI_DOUBLE,rank-1,0,comm->communicator,MPI_STATUS_IGNORE);
			MPI_Send(real_cell_first,count,MPI_DOUBLE,rank-1,0,comm->communicator);
		}
	}
	//communication to the right : 
	if(rank<comm_size-1){
		if(rank%2!=0){
			MPI_Send(real_cell_last,count,MPI_DOUBLE,rank+1,0,comm->communicator);
			MPI_Recv(ghost_cell_right,count,MPI_DOUBLE,rank+1,0,comm->communicator,MPI_STATUS_IGNORE);
		} 
		else{
			MPI_Recv(ghost_cell_right,count,MPI_DOUBLE,rank+1,0,comm->communicator,MPI_STATUS_IGNORE);
			MPI_Send(real_cell_last,count,MPI_DOUBLE,rank+1,0,comm->communicator);
		}
	}

	
	
	

	
}
