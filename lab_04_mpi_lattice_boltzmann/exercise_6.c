/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

//////////////////////////////////////////////////////
//
// Goal: Implement 2D grid communication with non-blocking
//       messages.
//
// SUMMARY:
//     - 2D splitting along X and Y
//     - 8 neighbors communications
//     - MPI type for non contiguous cells
// NEW:
//     - Non-blocking communications
//
//////////////////////////////////////////////////////

/****************************************************/
#include "src/lbm_struct.h"
#include "src/exercises.h"

/****************************************************/
void lbm_comm_init_ex6(lbm_comm_t * comm, int total_width, int total_height)
{
	lbm_comm_init_ex5(comm, total_width, total_height);
}

/****************************************************/
void lbm_comm_release_ex6(lbm_comm_t * comm)
{
	lbm_comm_release_ex5(comm);
}

/****************************************************/
void lbm_comm_ghost_exchange_ex6(lbm_comm_t * comm, lbm_mesh_t * mesh)
{
	int rank_up = comm->rank_y == 0 ? MPI_PROC_NULL : MPI_Cart_rank_wrapper(comm, comm->rank_x, comm->rank_y - 1);

	int rank_up_right = comm->rank_y == 0 || comm->rank_x == comm->nb_x - 1 ? MPI_PROC_NULL : MPI_Cart_rank_wrapper(comm, comm->rank_x + 1, comm->rank_y - 1);

	int rank_right = comm->rank_x == comm->nb_x - 1 ? MPI_PROC_NULL : MPI_Cart_rank_wrapper(comm, comm->rank_x + 1, comm->rank_y);

	int rank_right_down = comm->rank_y == comm->nb_y - 1 || comm->rank_x == comm->nb_x - 1 ? MPI_PROC_NULL : MPI_Cart_rank_wrapper(comm, comm->rank_x + 1, comm->rank_y + 1); 

	int rank_down = comm->rank_y == comm->nb_y - 1 ? MPI_PROC_NULL : MPI_Cart_rank_wrapper(comm, comm->rank_x, comm->rank_y + 1);

	int rank_down_left = comm->rank_y == comm->nb_y - 1 || comm->rank_x == 0 ? MPI_PROC_NULL : MPI_Cart_rank_wrapper(comm, comm->rank_x - 1, comm->rank_y + 1);

	int rank_left = comm->rank_x == 0 ? MPI_PROC_NULL : MPI_Cart_rank_wrapper(comm, comm->rank_x - 1, comm->rank_y);

	int rank_left_up = comm->rank_x == 0 || comm->rank_y == 0 ? MPI_PROC_NULL : MPI_Cart_rank_wrapper(comm, comm->rank_x - 1, comm->rank_y - 1);

	int count_data_vertically = comm->height * DIRECTIONS;
	
	//right 
	MPI_Irecv(lbm_mesh_get_cell(mesh,0,0), count_data_vertically, MPI_DOUBLE, rank_left, TAG, comm->communicator, &comm->requests[comm->req_count++]);
	MPI_Isend(lbm_mesh_get_cell(mesh,comm->width-2, 0), count_data_vertically, MPI_DOUBLE, rank_right, TAG, comm->communicator, &comm->requests[comm->req_count++]);
	
	//left 
	MPI_Irecv(lbm_mesh_get_cell(mesh,comm->width-1,0), count_data_vertically, MPI_DOUBLE, rank_right, TAG, comm->communicator, &comm->requests[comm->req_count++]);
	MPI_Isend(lbm_mesh_get_cell(mesh,1,0), count_data_vertically, MPI_DOUBLE, rank_left, TAG, comm->communicator, &comm->requests[comm->req_count++]);

	//up
	MPI_Irecv(lbm_mesh_get_cell(mesh, 0, comm->height - 1), 1, comm->type, rank_down, TAG, comm->communicator, &comm->requests[comm->req_count++]);
	MPI_Isend(lbm_mesh_get_cell(mesh, 0, 1), 1, comm->type, rank_up, TAG, comm->communicator, &comm->requests[comm->req_count++]);
	
	//down
	MPI_Irecv(lbm_mesh_get_cell(mesh, 0, 0), 1, comm->type, rank_up, TAG, comm->communicator, &comm->requests[comm->req_count++]);
	MPI_Isend(lbm_mesh_get_cell(mesh, 0, comm->height - 2), 1, comm->type, rank_down, TAG, comm->communicator, &comm->requests[comm->req_count++]);
	
	if (comm->req_count > 0) MPI_Waitall(comm->req_count, comm->requests, MPI_STATUS_IGNORE); // waiting for first batch to end.
	comm->req_count = 0;
	 

	//right diagonals
	MPI_Irecv(lbm_mesh_get_cell(mesh, comm->width - 1, 0), DIRECTIONS, MPI_DOUBLE, rank_up_right, TAG, comm->communicator, &comm->requests[comm->req_count++]);
	MPI_Isend(lbm_mesh_get_cell(mesh, comm->width - 2, 1), DIRECTIONS, MPI_DOUBLE, rank_up_right, TAG, comm->communicator, &comm->requests[comm->req_count++]);

	MPI_Irecv(lbm_mesh_get_cell(mesh, comm->width - 1, comm->height - 1), DIRECTIONS, MPI_DOUBLE, rank_right_down, TAG, comm->communicator, &comm->requests[comm->req_count++]); 
	MPI_Isend(lbm_mesh_get_cell(mesh, comm->width - 2, comm->height - 2), DIRECTIONS, MPI_DOUBLE, rank_right_down, TAG, comm->communicator, &comm->requests[comm->req_count++]);

	//left diagonals
	MPI_Irecv(lbm_mesh_get_cell(mesh, 0, 0), DIRECTIONS, MPI_DOUBLE, rank_left_up, TAG, comm->communicator, &comm->requests[comm->req_count++]); 
	MPI_Isend(lbm_mesh_get_cell(mesh, 1, 1), DIRECTIONS, MPI_DOUBLE, rank_left_up, TAG, comm->communicator, &comm->requests[comm->req_count++]);

	MPI_Irecv(lbm_mesh_get_cell(mesh, 0, comm->height - 1), DIRECTIONS, MPI_DOUBLE, rank_down_left, TAG, comm->communicator, &comm->requests[comm->req_count++]); 
	MPI_Isend(lbm_mesh_get_cell(mesh, 1, comm->height - 2), DIRECTIONS, MPI_DOUBLE, rank_down_left, TAG, comm->communicator, &comm->requests[comm->req_count++]);

	if (comm->req_count > 0) MPI_Waitall(comm->req_count, comm->requests, MPI_STATUS_IGNORE); //idem, for the second batch.
	comm->req_count = 0;
}
