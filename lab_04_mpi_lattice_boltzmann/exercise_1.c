/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

//////////////////////////////////////////////////////
//
//
// GOAL: Implement a 1D communication scheme along
//       X axis with blocking communications.
//
// SUMMARY:
//     - 1D splitting along X
//     - Blocking communications
//
//////////////////////////////////////////////////////

/****************************************************/
#include "src/lbm_struct.h"
#include "src/exercises.h"

/****************************************************/
void lbm_comm_init_ex1(lbm_comm_t * comm, int total_width, int total_height)
{
	
	int rank, comm_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	comm->communicator = MPI_COMM_WORLD;
	// vertical split.
	comm->nb_x = comm_size;
	comm->nb_y = 1;

	// rank (rank,0) in the grid
	comm->rank_x = rank;
	comm->rank_y = 0;

	
	comm->width = total_width/comm_size+2;
	comm->height = total_height+2; // why + 2 here? We are propagating along the x axis.

	comm->x = rank*(total_width/comm_size);
	comm->y = 0;

	lbm_comm_print(comm);
}

/****************************************************/
void lbm_comm_ghost_exchange_ex1(lbm_comm_t * comm, lbm_mesh_t * mesh)
{
	

	int rank, comm_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	int count = comm->height * DIRECTIONS;

	double * real_cell_first=lbm_mesh_get_cell(mesh,1,0);
	double * real_cell_last=lbm_mesh_get_cell(mesh,comm->width-2,0);

	double * ghost_cell_left=lbm_mesh_get_cell(mesh,0,0);
	double * ghost_cell_right=lbm_mesh_get_cell(mesh,comm->width-1,0);

	//communication in increasing order of ranks to the right :
	if(rank>0){
		MPI_Recv(ghost_cell_left,count,MPI_DOUBLE,rank-1,0,comm->communicator,MPI_STATUS_IGNORE);
	} 
	if(rank< comm_size-1){
		MPI_Send(real_cell_last,count,MPI_DOUBLE,rank+1,0,comm->communicator);
	}
	

	//communication to the left : 

	if(rank<comm_size-1){
		MPI_Recv(ghost_cell_right,count,MPI_DOUBLE,rank+1,1,comm->communicator,MPI_STATUS_IGNORE);
	} 
	if(rank>0){
		MPI_Send(real_cell_first,count,MPI_DOUBLE,rank-1,1,comm->communicator);
	}



}
