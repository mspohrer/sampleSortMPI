#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define max(a,b) ((a > b) ? a : b)
#define whichN ((rank < P-1) ? N:N+Nmod)

int rank, P, N, Nmod, pos, ns, buf_size, rtot;
int *as, *s, *buf, *rbuf;

// Bubble sort used in lab8?
//
void bubbleSort(int n, int a[]) {
  for (int i = 0; i < n; i++)
    for (int j = i+1; j < n; j++) 
      if (a[i] > a[j]) {
        int tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
      }
}

// opens the file in each process, gets the size of the file in bytes,
// sets each process to read in size/P amount of the file with the highest
// ranking process finishing off the last bit.
//
void read_file(char * infile) {
  MPI_File fin;
  MPI_Offset fsize;
  MPI_Status st;

  MPI_File_open(MPI_COMM_WORLD, infile, MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
  if(!fin) {
    printf("couldn't open file to read\n");
    MPI_Finalize();
    exit(0);
  }

  MPI_File_set_atomicity(fin, 1);
  MPI_File_get_size(fin, &fsize);

  int numI = fsize/4; 
  N = numI/(P);
  Nmod = numI%(P);
  pos = rank * N *sizeof(int);

  buf_size = whichN * sizeof(int);
  buf = malloc(buf_size);
  memset(buf,0,buf_size);

  MPI_File_read_at(fin, pos, buf, whichN, MPI_INT, &st);

  MPI_File_close(&fin);
}

// determines the local splitter, sends them to rank=0, 
// 0 sorts, then sends the list of sample parameters to 
// each process
//
void splitters() {
  ns = P * (P-1);
  as = malloc(sizeof(int) * (ns));
  memset(as,0,sizeof(int) * ns);
  s = malloc(sizeof(int) * (P-1));
  memset(s,0,sizeof(int) * (P-1));
  int step = (whichN/(P));

  for(int i=1; i<P; i++)
    s[i-1] = buf[(i*step)-1];

  // in root, sort all of the splitters sent to it by the other 
  // processes, and finds the universal splitters
  MPI_Gather(s, P-1, MPI_INT, as, P-1, MPI_INT, 0, MPI_COMM_WORLD);


  // gather all splitters from the other processes.

  if(rank == 0){
    bubbleSort(ns, as);
    step = ns/(P);
    for(int i=1; i<P; i++)
      s[i-1] = as[(i*step)-1];
  }

  MPI_Bcast(s,P-1,MPI_INT,0,MPI_COMM_WORLD);
}

// Uses the global splitters to bucketize the data. Originally
// I had a 2D array to hold the "samples" and it worked great 
// until I had to use alltoallv. I got the scatterv to work 
// but really wanted to use alltoallv, so I wrenched it in there
//
void split_send() {
  int i,j,k,scount[P], rcount[P], sdispl[P], rdispl[P];

  memset(scount, 0, P*sizeof(int));

  i=0;
  for(k=0; k<P-1; k++) {
    while(buf[i]<s[k]){
      ++scount[k];
      ++i;
      if(i>=whichN) break;
    }
    if(i>=whichN) break;
  }
  while(i<whichN) {
    ++scount[P-1];
    ++i;
    }

  MPI_Alltoall(scount, 1, MPI_INT, rcount, 1, MPI_INT, MPI_COMM_WORLD);

  rtot = 0;
  for(i=0; i<P; i++)
    rtot+=rcount[i];

  sdispl[0] = 0;
  rdispl[0] = 0;
  for(i = 1; i < P; i++) {
    sdispl[i] = scount[i-1] + sdispl[i-1];
    rdispl[i] = rdispl[i-1] + rcount[i-1];
  }

  rbuf = malloc(rtot * sizeof(int));
  memset(rbuf, 0, rtot * sizeof(int));

  MPI_Alltoallv(buf, scount, sdispl, MPI_INT, rbuf, rcount, rdispl, MPI_INT, MPI_COMM_WORLD);
}

void free_all() {
  free(s);
  free(as);
  free(buf);
  free(rbuf);
}

void write_to_file(char * outfile) {
  MPI_File fout;
  MPI_Offset fsize;
  MPI_Status st;
  int displ[P],r[P];

  MPI_File_delete(outfile, MPI_INFO_NULL);
  MPI_File_open(MPI_COMM_WORLD, outfile, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fout);
  MPI_File_set_atomicity(fout, 1);
  if(!fout) {
    printf("couldn't open file to read\n");
    MPI_Finalize();
    exit(0);
  }

  MPI_Allgather(&rtot, 1, MPI_INT, r, 1, MPI_INT, MPI_COMM_WORLD);

  displ[0]=0;
  for(int i = 0; i < P; i++) 
    displ[i+1] = displ[i] + r[i];
  /*
  printf("rank: %d, whichN %di rtot %d:", rank, whichN, rtot);
  for(int i = 0; i < whichN; ++i){
    printf(" %d", buf[i]);
  }
  printf("\n");
  */
  MPI_File_write_at(fout,sizeof(int)* displ[rank], rbuf, rtot, MPI_INT, &st);
  MPI_File_close(&fout);
}

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);  

  if (argc < 3) {
    printf("Usage: ./sampleSort <infile> <outfile>\n");
    MPI_Finalize();
    exit(0);
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
  MPI_Comm_size(MPI_COMM_WORLD, &P);

  printf("%d starting %s\n", rank, argv[1]);

  read_file(argv[1]);

  bubbleSort(whichN, buf);

  splitters();
  split_send();
  
  bubbleSort(rtot, rbuf);
/*
  printf(6rank: %d:", rank);
  int k = 0;
  for(int i = 0; i < rtot; ++i){
    printf(" %d", rbuf[i]);
    ++k;
  }
  printf(" k:%d\n", k);
*/
  write_to_file(argv[2]);

  printf("%d exiting\n", rank);
  free_all();
  MPI_Finalize();
}
