#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "..\usr\include\GL\freeglut.h"

#define DIM 1024
#define TILE_WIDTH 32
#define SPHERES 17
#define PLANES 8
#define MAX(x, y)	((x) > (y) ? (x) : (y))
#define SQR(x)	((x) * (x))
#define ABS(x)	(((x) > 0.0)? (x):(-x))

typedef float Vec3[3];

// 3차원 구를 정의하는 구조체
struct Sphere {
	Vec3	Pos;	// 구의 중심 좌표
	float	Rad;	// 구의 반지름
	Vec3	Ka;
	Vec3	Kd;		// 난반사 계수
	Vec3	Ks;		// 전반사 계수
	float	ns;		// 전반사 지수
	bool bTransparent;//불투명 여부
};
//평면 정의하는 구조체
struct Plane {
	Vec3	p1;		//평면 꼭짓점1
	Vec3	p2;		//평면 꼭짓점2
	Vec3	p3;		//평면 꼭짓점3
	Vec3	p4;		//평면 꼭짓점4
	Vec3	Ka;
	Vec3	Kd;		// 난반사 계수
	Vec3	Ks;		// 전반사 계수
	float	ns;		// 전반사 지수
	Vec3	n;		//법선
};

unsigned char pHostSrcImage[DIM * DIM * 3];
unsigned char pHostOutImage[DIM * DIM];
unsigned char pHostGrayResult[DIM * DIM];
unsigned char pHostTempImage[DIM*DIM];
float *pHostSobelResult;

unsigned char *pDevSrcImage = NULL;
unsigned char *pDevOutImage = NULL;
unsigned char *pDevGrayResult = NULL;
unsigned char *pDevTempImage = NULL;
float *pDevSobelResult = NULL;

int keynum = 0;

Sphere *SphereList;
Plane *PlaneList;

__global__ void RayTrace_Kernel(Sphere *SphereList, Plane *Planes, unsigned char *pImage, unsigned char *pDevGrayResult);
__global__ void SobelEdge(unsigned char *pDevSrc, float *pDevSobelResult, int S);
__global__ void SetEdgeColor(unsigned char *pDevOut, float *pDevSobelResult, int min, int max);
__global__ void Gaussian(unsigned char *pDevSrc, unsigned char *pDevOut, int S);
__global__ void Median(unsigned char *pDevSrc, unsigned char *pDevOut, unsigned char *pTempImage, int S);
__global__ void SetMedianResult(unsigned char *pDevOut, unsigned char *pTempImage);
__device__ void Phong(Sphere *Spheres, Plane *Planes, Vec3 P, Vec3 N, Sphere S, Vec3 C, float k);
__device__ void PhongP(Sphere *Spheres, Plane *Planes, Vec3 P, Vec3 N, Plane PL, Vec3 C, float k);
__device__ bool intersect_line_sphere(Sphere *Spheres, Vec3 p, Vec3 v, int &sidx, float &t);
__device__ bool intersect_line_plane(Plane *Planes, Vec3 p, Vec3 v, int &pidx, float &t);
__device__ void Sub(Vec3 c, Vec3 a, Vec3 b);
__device__ float Dot(Vec3 a, Vec3 b);
__device__ void Normalize(Vec3 a);
__device__ void Line(Vec3 pt, Vec3 pos, Vec3 dir, float time);
__device__ void Outer(Vec3 c, Vec3 a, Vec3 b);

// 콜백 함수 선언
void Render();
void Reshape(int w, int h);
void Timer(int id);
void Close();
void InitScene();
void CreateImage();
void Keyboard(unsigned char key, int x, int y);

int main(int argc, char **argv)
{
	// OpenGL 초기화, 윈도우 크기 설정, 디스플레이 모드 설정
	glutInit(&argc, argv);
	glutInitWindowSize(DIM, DIM);
	glutInitDisplayMode(GLUT_RGB);

	// 윈도우 생성 및 콜백 함수 등록
	glutCreateWindow("RayTracer on GPU");
	glutDisplayFunc(Render);
	glutReshapeFunc(Reshape);
	glutKeyboardFunc(Keyboard);
	glutTimerFunc(10, Timer, 0);
	glutCloseFunc(Close);

	InitScene();
	CreateImage();
	// 이벤트를 처리를 위한 무한 루프로 진입한다.
	glutMainLoop();

	delete[] pHostSrcImage;
	delete[] pHostOutImage;
	delete[] pHostSobelResult;
	delete[] pHostGrayResult;
	return 0;
}

void InitScene()
{
	cudaSetDevice(0);
	cudaMalloc((void **)&pDevGrayResult, sizeof(unsigned char) * DIM * DIM);
	cudaMalloc((void **)&pDevSrcImage, sizeof(unsigned char) * DIM * DIM * 3);
	cudaMalloc((void **)&pDevOutImage, sizeof(unsigned char) * DIM * DIM);
	cudaMalloc((void **)&pDevTempImage, sizeof(unsigned char) * DIM * DIM);
	cudaMalloc((void **)&pDevSobelResult, sizeof(float) * DIM * DIM);

	cudaMalloc((void **)&SphereList, sizeof(Sphere) * SPHERES);
	cudaMalloc((void **)&PlaneList, sizeof(Plane) * PLANES);
	pHostSobelResult = new float[DIM*DIM];

	memset(pHostSobelResult, 0, sizeof(float)*DIM*DIM);

	Sphere *temp_s = new Sphere[SPHERES];
	Plane *temp_p = new Plane[PLANES];

	// 장면에 구를 배치한다.
	temp_s[0].Pos[0] = 0.0;
	temp_s[0].Pos[1] = 0.0;
	temp_s[0].Pos[2] = -400.0;
	temp_s[0].Rad = 35.0;
	temp_s[0].Ka[0] = 0.1;
	temp_s[0].Ka[1] = 0.1;
	temp_s[0].Ka[2] = 0.1;
	temp_s[0].Kd[0] = 0.8;
	temp_s[0].Kd[1] = 0.8;
	temp_s[0].Kd[2] = 0.8;
	temp_s[0].Ks[0] = 0.9;
	temp_s[0].Ks[1] = 0.9;
	temp_s[0].Ks[2] = 0.9;
	temp_s[0].ns = 8.0;
	temp_s[0].bTransparent = true;

	temp_s[1].Pos[0] = -8.0;
	temp_s[1].Pos[1] = -13.0;
	temp_s[1].Pos[2] = -585.0;
	temp_s[1].Rad = 5.0;
	temp_s[1].Ka[0] = 0.2;
	temp_s[1].Ka[1] = 0.2;
	temp_s[1].Ka[2] = 0.2;
	temp_s[1].Kd[0] = 0.7;
	temp_s[1].Kd[1] = 0.5;
	temp_s[1].Kd[2] = 0.0;
	temp_s[1].Ks[0] = 0.8;
	temp_s[1].Ks[1] = 0.8;
	temp_s[1].Ks[2] = 0.8;
	temp_s[1].ns = 8.0;
	temp_s[1].bTransparent = true;//왼쪽 아래 빨강

	temp_s[2].Pos[0] = -40.0;
	temp_s[2].Pos[1] = -20.0;
	temp_s[2].Pos[2] = -700.0;
	temp_s[2].Rad = 24.0;
	temp_s[2].Ka[0] = 0.2;
	temp_s[2].Ka[1] = 0.2;
	temp_s[2].Ka[2] = 0.2;
	temp_s[2].Kd[0] = 0.0;
	temp_s[2].Kd[1] = 0.7;
	temp_s[2].Kd[2] = 0.0;
	temp_s[2].Ks[0] = 0.8;
	temp_s[2].Ks[1] = 0.8;
	temp_s[2].Ks[2] = 0.8;
	temp_s[2].ns = 8.0;
	temp_s[2].bTransparent = false;//왼쪽 아래 초록

	temp_s[3].Pos[0] = -35.0;
	temp_s[3].Pos[1] = -30.0;
	temp_s[3].Pos[2] = -620.0;
	temp_s[3].Rad = 20.0;
	temp_s[3].Ka[0] = 0.2;
	temp_s[3].Ka[1] = 0.2;
	temp_s[3].Ka[2] = 0.2;
	temp_s[3].Kd[0] = 0.0;
	temp_s[3].Kd[1] = 0.6;
	temp_s[3].Kd[2] = 0.3;
	temp_s[3].Ks[0] = 0.8;
	temp_s[3].Ks[1] = 0.8;
	temp_s[3].Ks[2] = 0.8;
	temp_s[3].ns = 8.0;
	temp_s[3].bTransparent = false;//오른쪽 위 초록

	temp_s[4].Pos[0] = 40.0;
	temp_s[4].Pos[1] = -25.0;
	temp_s[4].Pos[2] = -650.0;
	temp_s[4].Rad = 20.0;
	temp_s[4].Ka[0] = 0.2;
	temp_s[4].Ka[1] = 0.2;
	temp_s[4].Ka[2] = 0.2;
	temp_s[4].Kd[0] = 0.0;
	temp_s[4].Kd[1] = 0.9;
	temp_s[4].Kd[2] = 0.5;
	temp_s[4].Ks[0] = 0.8;
	temp_s[4].Ks[1] = 0.8;
	temp_s[4].Ks[2] = 0.8;
	temp_s[4].ns = 8.0;
	temp_s[4].bTransparent = false;//오른쪽 아래 초록

	temp_s[5].Pos[0] = 0.0;
	temp_s[5].Pos[1] = 0.0;
	temp_s[5].Pos[2] = -650.0;
	temp_s[5].Rad = 30.0;
	temp_s[5].Ka[0] = 0.5;
	temp_s[5].Ka[1] = 0.5;
	temp_s[5].Ka[2] = 0.5;
	temp_s[5].Kd[0] = 0.0;
	temp_s[5].Kd[1] = 0.5;
	temp_s[5].Kd[2] = 0.0;
	temp_s[5].Ks[0] = 0.9;
	temp_s[5].Ks[1] = 0.9;
	temp_s[5].Ks[2] = 0.9;
	temp_s[5].ns = 8.0;
	temp_s[5].bTransparent = false;//중심 초록

	temp_s[6].Pos[0] = -10.0;
	temp_s[6].Pos[1] = 40.0;
	temp_s[6].Pos[2] = -650.0;
	temp_s[6].Rad = 15.0;
	temp_s[6].Ka[0] = 0.5;
	temp_s[6].Ka[1] = 0.5;
	temp_s[6].Ka[2] = 0.5;
	temp_s[6].Kd[0] = 0.0;
	temp_s[6].Kd[1] = 0.3;
	temp_s[6].Kd[2] = 0.0;
	temp_s[6].Ks[0] = 0.9;
	temp_s[6].Ks[1] = 0.9;
	temp_s[6].Ks[2] = 0.9;
	temp_s[6].ns = 8.0;
	temp_s[6].bTransparent = false;//뒤 위 작은 초록

	temp_s[7].Pos[0] = -25.0;
	temp_s[7].Pos[1] = 5.0;
	temp_s[7].Pos[2] = -620.0;
	temp_s[7].Rad = 13.0;
	temp_s[7].Ka[0] = 0.2;
	temp_s[7].Ka[1] = 0.2;
	temp_s[7].Ka[2] = 0.2;
	temp_s[7].Kd[0] = 0.3;
	temp_s[7].Kd[1] = 0.7;
	temp_s[7].Kd[2] = 0.0;
	temp_s[7].Ks[0] = 0.8;
	temp_s[7].Ks[1] = 0.8;
	temp_s[7].Ks[2] = 0.8;
	temp_s[7].ns = 8.0;
	temp_s[7].bTransparent = false;//왼쪽 위 초록

	temp_s[8].Pos[0] = 29.0;
	temp_s[8].Pos[1] = -30.0;
	temp_s[8].Pos[2] = -630.0;
	temp_s[8].Rad = 13.0;
	temp_s[8].Ka[0] = 0.2;
	temp_s[8].Ka[1] = 0.2;
	temp_s[8].Ka[2] = 0.2;
	temp_s[8].Kd[0] = 0.3;
	temp_s[8].Kd[1] = 0.7;
	temp_s[8].Kd[2] = 0.0;
	temp_s[8].Ks[0] = 0.8;
	temp_s[8].Ks[1] = 0.8;
	temp_s[8].Ks[2] = 0.8;
	temp_s[8].ns = 8.0;
	temp_s[8].bTransparent = false;//앞 아래 초록

	temp_s[9].Pos[0] = 35.0;
	temp_s[9].Pos[1] = 20.0;
	temp_s[9].Pos[2] = -670.0;
	temp_s[9].Rad = 25.0;
	temp_s[9].Ka[0] = 0.5;
	temp_s[9].Ka[1] = 0.5;
	temp_s[9].Ka[2] = 0.5;
	temp_s[9].Kd[0] = 0.0;
	temp_s[9].Kd[1] = 0.3;
	temp_s[9].Kd[2] = 0.0;
	temp_s[9].Ks[0] = 0.9;
	temp_s[9].Ks[1] = 0.9;
	temp_s[9].Ks[2] = 0.9;
	temp_s[9].ns = 8.0;
	temp_s[9].bTransparent = false;//오른쪽 뒤 초록

	temp_s[10].Pos[0] = -15.0;
	temp_s[10].Pos[1] = 25.0;
	temp_s[10].Pos[2] = -630.0;
	temp_s[10].Rad = 4.0;
	temp_s[10].Ka[0] = 0.2;
	temp_s[10].Ka[1] = 0.2;
	temp_s[10].Ka[2] = 0.2;
	temp_s[10].Kd[0] = 0.7;
	temp_s[10].Kd[1] = 0.2;
	temp_s[10].Kd[2] = 0.0;
	temp_s[10].Ks[0] = 0.8;
	temp_s[10].Ks[1] = 0.8;
	temp_s[10].Ks[2] = 0.8;
	temp_s[10].ns = 8.0;
	temp_s[10].bTransparent = false;//왼쪽 위 빨강

	temp_s[11].Pos[0] = 32.0;
	temp_s[11].Pos[1] = -10.0;
	temp_s[11].Pos[2] = -620.0;
	temp_s[11].Rad = 3.0;
	temp_s[11].Ka[0] = 0.2;
	temp_s[11].Ka[1] = 0.2;
	temp_s[11].Ka[2] = 0.2;
	temp_s[11].Kd[0] = 0.7;
	temp_s[11].Kd[1] = 0.2;
	temp_s[11].Kd[2] = 0.0;
	temp_s[11].Ks[0] = 0.8;
	temp_s[11].Ks[1] = 0.8;
	temp_s[11].Ks[2] = 0.8;
	temp_s[11].ns = 8.0;
	temp_s[11].bTransparent = false;//오른쪽 아래 빨강

	temp_s[12].Pos[0] = -30.0;
	temp_s[12].Pos[1] = 25.0;
	temp_s[12].Pos[2] = -680.0;
	temp_s[12].Rad = 22.0;
	temp_s[12].Ka[0] = 0.5;
	temp_s[12].Ka[1] = 0.5;
	temp_s[12].Ka[2] = 0.5;
	temp_s[12].Kd[0] = 0.0;
	temp_s[12].Kd[1] = 0.2;
	temp_s[12].Kd[2] = 0.0;
	temp_s[12].Ks[0] = 0.9;
	temp_s[12].Ks[1] = 0.9;
	temp_s[12].Ks[2] = 0.9;
	temp_s[12].ns = 8.0;
	temp_s[12].bTransparent = false;//뒤 위 작은 초록

	temp_s[13].Pos[0] = -3.0;
	temp_s[13].Pos[1] = -25.0;
	temp_s[13].Pos[2] = -610.0;
	temp_s[13].Rad = 9.0;
	temp_s[13].Ka[0] = 0.5;
	temp_s[13].Ka[1] = 0.5;
	temp_s[13].Ka[2] = 0.5;
	temp_s[13].Kd[0] = 0.0;
	temp_s[13].Kd[1] = 0.2;
	temp_s[13].Kd[2] = 0.0;
	temp_s[13].Ks[0] = 0.9;
	temp_s[13].Ks[1] = 0.9;
	temp_s[13].Ks[2] = 0.9;
	temp_s[13].ns = 8.0;
	temp_s[13].bTransparent = false;//왼쪽 아래 어두운 초록

	temp_s[14].Pos[0] = 150.0;
	temp_s[14].Pos[1] = 100.0;
	temp_s[14].Pos[2] = -500.0;
	temp_s[14].Rad = 7.0;
	temp_s[14].Ka[0] = 0.1;
	temp_s[14].Ka[1] = 0.1;
	temp_s[14].Ka[2] = 0.1;
	temp_s[14].Kd[0] = 0.5;
	temp_s[14].Kd[1] = 0.5;
	temp_s[14].Kd[2] = 0.7;
	temp_s[14].Ks[0] = 0.9;
	temp_s[14].Ks[1] = 0.9;
	temp_s[14].Ks[2] = 0.9;
	temp_s[14].ns = 8.0;
	temp_s[14].bTransparent = true;//눈1

	temp_s[15].Pos[0] = 30.0;
	temp_s[15].Pos[1] = 100.0;
	temp_s[15].Pos[2] = -520.0;
	temp_s[15].Rad = 5.0;
	temp_s[15].Ka[0] = 0.1;
	temp_s[15].Ka[1] = 0.1;
	temp_s[15].Ka[2] = 0.1;
	temp_s[15].Kd[0] = 0.7;
	temp_s[15].Kd[1] = 0.7;
	temp_s[15].Kd[2] = 1.0;
	temp_s[15].Ks[0] = 0.9;
	temp_s[15].Ks[1] = 0.9;
	temp_s[15].Ks[2] = 0.9;
	temp_s[15].ns = 8.0;
	temp_s[15].bTransparent = true;//눈2

	temp_s[16].Pos[0] = 0.0;
	temp_s[16].Pos[1] = 80.0;
	temp_s[16].Pos[2] = -600.0;
	temp_s[16].Rad = 5.0;
	temp_s[16].Ka[0] = 0.1;
	temp_s[16].Ka[1] = 0.1;
	temp_s[16].Ka[2] = 0.1;
	temp_s[16].Kd[0] = 0.7;
	temp_s[16].Kd[1] = 0.7;
	temp_s[16].Kd[2] = 1.0;
	temp_s[16].Ks[0] = 0.9;
	temp_s[16].Ks[1] = 0.9;
	temp_s[16].Ks[2] = 0.9;
	temp_s[16].ns = 8.0;
	temp_s[16].bTransparent = true;//눈3

	//맨 뒤
	temp_p[0].p1[0] = -20.0;
	temp_p[0].p1[1] = -100.0;
	temp_p[0].p1[2] = -700.0;
	temp_p[0].p2[0] = 20.0;
	temp_p[0].p2[1] = -100.0;
	temp_p[0].p2[2] = -700.0;
	temp_p[0].p3[0] = 20.0;
	temp_p[0].p3[1] = -10.0;
	temp_p[0].p3[2] = -700.0;
	temp_p[0].p4[0] = -20.0;
	temp_p[0].p4[1] = -10.0;
	temp_p[0].p4[2] = -700.0;
	temp_p[0].Ka[0] = 0.2;
	temp_p[0].Ka[1] = 0.2;
	temp_p[0].Ka[2] = 0.2;
	temp_p[0].Kd[0] = 0.3;
	temp_p[0].Kd[1] = 0.2;
	temp_p[0].Kd[2] = 0.1;
	temp_p[0].Ks[0] = 0.8;
	temp_p[0].Ks[1] = 0.8;
	temp_p[0].Ks[2] = 0.8;
	temp_p[0].ns = 8.0;

	//오른쪽 옆면
	temp_p[1].p1[0] = 20.0;
	temp_p[1].p1[1] = -100.0;
	temp_p[1].p1[2] = -650.0;
	temp_p[1].p2[0] = 20.0;
	temp_p[1].p2[1] = -100.0;
	temp_p[1].p2[2] = -700.0;
	temp_p[1].p3[0] = 20.0;
	temp_p[1].p3[1] = -10.0;
	temp_p[1].p3[2] = -700.0;
	temp_p[1].p4[0] = 20.0;
	temp_p[1].p4[1] = -10.0;
	temp_p[1].p4[2] = -650.0;
	temp_p[1].Ka[0] = 0.2;
	temp_p[1].Ka[1] = 0.2;
	temp_p[1].Ka[2] = 0.2;
	temp_p[1].Kd[0] = 0.3;
	temp_p[1].Kd[1] = 0.2;
	temp_p[1].Kd[2] = 0.1;
	temp_p[1].Ks[0] = 0.8;
	temp_p[1].Ks[1] = 0.8;
	temp_p[1].Ks[2] = 0.8;
	temp_p[1].ns = 8.0;

	//앞면
	temp_p[2].p1[0] = -20.0;
	temp_p[2].p1[1] = -100.0;
	temp_p[2].p1[2] = -650.0;
	temp_p[2].p2[0] = 20.0;
	temp_p[2].p2[1] = -100.0;
	temp_p[2].p2[2] = -650.0;
	temp_p[2].p3[0] = 20.0;
	temp_p[2].p3[1] = -10.0;
	temp_p[2].p3[2] = -650.0;
	temp_p[2].p4[0] = -20.0;
	temp_p[2].p4[1] = -10.0;
	temp_p[2].p4[2] = -650.0;
	temp_p[2].Ka[0] = 0.2;
	temp_p[2].Ka[1] = 0.2;
	temp_p[2].Ka[2] = 0.2;
	temp_p[2].Kd[0] = 0.3;
	temp_p[2].Kd[1] = 0.2;
	temp_p[2].Kd[2] = 0.1;
	temp_p[2].Ks[0] = 0.8;
	temp_p[2].Ks[1] = 0.8;
	temp_p[2].Ks[2] = 0.8;
	temp_p[2].ns = 8.0;

	//왼쪽 옆면
	temp_p[3].p1[0] = -20.0;
	temp_p[3].p1[1] = -10.0;
	temp_p[3].p1[2] = -650.0;
	temp_p[3].p2[0] = -20.0;
	temp_p[3].p2[1] = -10.0;
	temp_p[3].p2[2] = -700.0;
	temp_p[3].p3[0] = -20.0;
	temp_p[3].p3[1] = -100.0;
	temp_p[3].p3[2] = -700.0;
	temp_p[3].p4[0] = -20.0;
	temp_p[3].p4[1] = -100.0;
	temp_p[3].p4[2] = -650.0;
	temp_p[3].Ka[0] = 0.2;
	temp_p[3].Ka[1] = 0.2;
	temp_p[3].Ka[2] = 0.2;
	temp_p[3].Kd[0] = 0.3;
	temp_p[3].Kd[1] = 0.2;
	temp_p[3].Kd[2] = 0.1;
	temp_p[3].Ks[0] = 0.8;
	temp_p[3].Ks[1] = 0.8;
	temp_p[3].Ks[2] = 0.8;
	temp_p[3].ns = 8.0;

	//아랫면
	temp_p[4].p1[0] = -20.0;
	temp_p[4].p1[1] = -100.0;
	temp_p[4].p1[2] = -650.0;
	temp_p[4].p2[0] = 20.0;
	temp_p[4].p2[1] = -100.0;
	temp_p[4].p2[2] = -650.0;
	temp_p[4].p3[0] = 20.0;
	temp_p[4].p3[1] = -100.0;
	temp_p[4].p3[2] = -700.0;
	temp_p[4].p4[0] = -20.0;
	temp_p[4].p4[1] = -100.0;
	temp_p[4].p4[2] = -700.0;
	temp_p[4].Ka[0] = 0.2;
	temp_p[4].Ka[1] = 0.2;
	temp_p[4].Ka[2] = 0.2;
	temp_p[4].Kd[0] = 0.3;
	temp_p[4].Kd[1] = 0.2;
	temp_p[4].Kd[2] = 0.1;
	temp_p[4].Ks[0] = 0.8;
	temp_p[4].Ks[1] = 0.8;
	temp_p[4].Ks[2] = 0.8;
	temp_p[4].ns = 8.0;

	//윗면
	temp_p[5].p1[0] = -20.0;
	temp_p[5].p1[1] = -10.0;
	temp_p[5].p1[2] = -650.0;
	temp_p[5].p2[0] = 20.0;
	temp_p[5].p2[1] = -10.0;
	temp_p[5].p2[2] = -650.0;
	temp_p[5].p3[0] = 20.0;
	temp_p[5].p3[1] = -10.0;
	temp_p[5].p3[2] = -700.0;
	temp_p[5].p4[0] = -20.0;
	temp_p[5].p4[1] = -10.0;
	temp_p[5].p4[2] = -700.0;
	temp_p[5].Ka[0] = 0.2;
	temp_p[5].Ka[1] = 0.2;
	temp_p[5].Ka[2] = 0.2;
	temp_p[5].Kd[0] = 0.3;
	temp_p[5].Kd[1] = 0.2;
	temp_p[5].Kd[2] = 0.1;
	temp_p[5].Ks[0] = 0.8;
	temp_p[5].Ks[1] = 0.8;
	temp_p[5].Ks[2] = 0.8;
	temp_p[5].ns = 8.0;

	//바닥
	temp_p[6].p1[0] = -150.0;
	temp_p[6].p1[1] = -100.0;
	temp_p[6].p1[2] = -350.0;
	temp_p[6].p2[0] = 250.0;
	temp_p[6].p2[1] = -100.0;
	temp_p[6].p2[2] = -350.0;
	temp_p[6].p3[0] = 250.0;
	temp_p[6].p3[1] = -100.0;
	temp_p[6].p3[2] = -850.0;
	temp_p[6].p4[0] = -150.0;
	temp_p[6].p4[1] = -100.0;
	temp_p[6].p4[2] = -850.0;
	temp_p[6].Ka[0] = 0.2;
	temp_p[6].Ka[1] = 0.2;
	temp_p[6].Ka[2] = 0.2;
	temp_p[6].Kd[0] = 0.4;
	temp_p[6].Kd[1] = 0.3;
	temp_p[6].Kd[2] = 0.1;
	temp_p[6].Ks[0] = 1.0;
	temp_p[6].Ks[1] = 1.0;
	temp_p[6].Ks[2] = 1.0;
	temp_p[6].ns = 8.0;

	//벽
	temp_p[7].p1[0] = -150.0;
	temp_p[7].p1[1] = -100.0;
	temp_p[7].p1[2] = -850.0;
	temp_p[7].p2[0] = 250.0;
	temp_p[7].p2[1] = -100.0;
	temp_p[7].p2[2] = -850.0;
	temp_p[7].p3[0] = 250.0;
	temp_p[7].p3[1] = 150.0;
	temp_p[7].p3[2] = -850.0;
	temp_p[7].p4[0] = -150.0;
	temp_p[7].p4[1] = 150.0;
	temp_p[7].p4[2] = -850.0;
	temp_p[7].Ka[0] = 0.2;
	temp_p[7].Ka[1] = 0.2;
	temp_p[7].Ka[2] = 0.2;
	temp_p[7].Kd[0] = 0.2;
	temp_p[7].Kd[1] = 0.7;
	temp_p[7].Kd[2] = 0.8;
	temp_p[7].Ks[0] = 1.0;
	temp_p[7].Ks[1] = 1.0;
	temp_p[7].Ks[2] = 1.0;
	temp_p[7].ns = 9.0;

	// GPU로 복사한다.
	cudaMemcpy(SphereList, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice);
	cudaMemcpy(PlaneList, temp_p, sizeof(Plane) * PLANES, cudaMemcpyHostToDevice);
	delete[] temp_s;
	delete[] temp_p;
}

void Reshape(int w, int h)
{
	glViewport(0, 0, w, h);
}

void Render()
{
	// 칼라 버퍼와 깊이 버퍼 지우기
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	if(keynum == 0)
		glDrawPixels(DIM, DIM, GL_RGB, GL_UNSIGNED_BYTE, pHostSrcImage);
	else if(keynum == 1)
		glDrawPixels(DIM, DIM, GL_LUMINANCE, GL_UNSIGNED_BYTE, pHostGrayResult);
	else
		glDrawPixels(DIM, DIM, GL_LUMINANCE, GL_UNSIGNED_BYTE, pHostOutImage);

	// 칼라 버퍼 교환한다
	glutSwapBuffers();
}

void Timer(int id)
{
	clock_t st = clock();
	static float theta = 0.0;
	theta += 0.05;
	float x = 110 * cos(theta);
	float z = 110 * sin(theta);

	Sphere s1, s2, s3, s4, s5, s6, s7;
	cudaMemcpy(&s1, &SphereList[0], sizeof(Sphere) * 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(&s2, &SphereList[1], sizeof(Sphere) * 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(&s3, &SphereList[10], sizeof(Sphere) * 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(&s4, &SphereList[11], sizeof(Sphere) * 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(&s5, &SphereList[14], sizeof(Sphere) * 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(&s6, &SphereList[15], sizeof(Sphere) * 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(&s7, &SphereList[16], sizeof(Sphere) * 1, cudaMemcpyDeviceToHost);
	s1.Pos[0] = x - 20;
	s1.Pos[2] = z - 650;

	if (s5.Pos[0] >= -44.0)
	{
		s5.Pos[0] -= 0.5;
		s5.Pos[1] -= 0.5;
		s5.Pos[2] -= 0.1;
	}
	if (s6.Pos[0] >= -18.7)
	{
		s6.Pos[0] -= 0.1;
		s6.Pos[1] -= 0.4;
		s6.Pos[2] -= 0.1;
	}
	if (s7.Pos[0] <= 43.7)
	{
		s7.Pos[0] += 0.1;
		s7.Pos[1] -= 0.4;
		s7.Pos[2] -= 0.01;
	}
	if (s2.Rad <= 8)
	{
		s2.Rad += 0.01;
		s2.Kd[0] += 0.0006;
		s2.Kd[1] -= 0.0006;
	}
	if (s3.Rad <= 7)
	{
		s3.Rad += 0.01;
		s3.Kd[0] += 0.0006;
		s3.Kd[1] -= 0.0006;
	}
	if (s4.Rad <= 9)
	{
		s4.Rad += 0.01;
		s4.Kd[0] += 0.0003;
		s4.Kd[1] -= 0.0003;
	}
	cudaMemcpy(&SphereList[0], &s1, sizeof(Sphere) * 1, cudaMemcpyHostToDevice);
	cudaMemcpy(&SphereList[1], &s2, sizeof(Sphere) * 1, cudaMemcpyHostToDevice);
	cudaMemcpy(&SphereList[10], &s3, sizeof(Sphere) * 1, cudaMemcpyHostToDevice);
	cudaMemcpy(&SphereList[11], &s4, sizeof(Sphere) * 1, cudaMemcpyHostToDevice);
	cudaMemcpy(&SphereList[14], &s5, sizeof(Sphere) * 1, cudaMemcpyHostToDevice);
	cudaMemcpy(&SphereList[15], &s6, sizeof(Sphere) * 1, cudaMemcpyHostToDevice);
	cudaMemcpy(&SphereList[16], &s7, sizeof(Sphere) * 1, cudaMemcpyHostToDevice);

	CreateImage();
	printf("Elapsed time = %u ms\n", clock() - st);
	glutPostRedisplay();
	glutTimerFunc(1, Timer, 0);
}

void CreateImage()
{
	dim3 gridDim(DIM / TILE_WIDTH, DIM / TILE_WIDTH);
	dim3 blockDim(TILE_WIDTH, TILE_WIDTH);

	cudaMemcpy(pDevGrayResult, pHostGrayResult, DIM*DIM * sizeof(unsigned char), cudaMemcpyHostToDevice);

	RayTrace_Kernel << <gridDim, blockDim >> >(SphereList, PlaneList, pDevSrcImage, pDevGrayResult);
	cudaDeviceSynchronize();

	cudaMemcpy(pHostSrcImage, pDevSrcImage, sizeof(unsigned char) * DIM * DIM * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(pHostGrayResult, pDevGrayResult, sizeof(unsigned char) * DIM * DIM, cudaMemcpyDeviceToHost);

	if (keynum == 2) {
		cudaMemcpy(pDevOutImage, pHostOutImage, DIM*DIM * sizeof(unsigned char), cudaMemcpyHostToDevice);
		cudaMemcpy(pDevSobelResult, pHostSobelResult, DIM*DIM * sizeof(unsigned char), cudaMemcpyHostToDevice);

		SobelEdge << <gridDim, blockDim >> > (pDevGrayResult, pDevSobelResult, DIM);

		cudaDeviceSynchronize();
		cudaMemcpy(pHostSobelResult, pDevSobelResult, sizeof(float)*DIM*DIM, cudaMemcpyDeviceToHost);

		float min = 1.0e+10, max = -1.0e+10;
		for (int i = 1; i < DIM - 1; i++)
		{
			for (int j = 1; j < DIM - 1; j++)
			{
				int idx = i * DIM + j;

				min = (pHostSobelResult[idx] < min) ? pHostSobelResult[idx] : min;
				max = (pHostSobelResult[idx] > max) ? pHostSobelResult[idx] : max;
			}
		}
		SetEdgeColor << <gridDim, blockDim >> > (pDevOutImage, pDevSobelResult, min, max);
		cudaDeviceSynchronize();
		cudaMemcpy(pHostOutImage, pDevOutImage, sizeof(unsigned char)*DIM*DIM, cudaMemcpyDeviceToHost);
	}
	else if (keynum == 3) {
		cudaMemcpy(pDevOutImage, pHostOutImage, DIM*DIM * sizeof(unsigned char), cudaMemcpyHostToDevice);

		Gaussian << <gridDim, blockDim >> > (pDevGrayResult, pDevOutImage, DIM);
		cudaDeviceSynchronize();
		cudaMemcpy(pHostOutImage, pDevOutImage, sizeof(unsigned char)*DIM*DIM, cudaMemcpyDeviceToHost);
	}
	else if (keynum == 4) {
		cudaMemcpy(pDevOutImage, pHostOutImage, DIM*DIM * sizeof(unsigned char), cudaMemcpyHostToDevice);
		cudaMemcpy(pDevOutImage, pHostTempImage, DIM*DIM * sizeof(unsigned char), cudaMemcpyHostToDevice);

		Median << <gridDim, blockDim >> > (pDevGrayResult, pDevOutImage, pDevTempImage, DIM);

		cudaDeviceSynchronize();
		cudaMemcpy(pHostTempImage, pDevTempImage, sizeof(unsigned char)*DIM*DIM, cudaMemcpyDeviceToHost);

		SetMedianResult << <gridDim, blockDim >> > (pDevOutImage, pDevTempImage);
		cudaDeviceSynchronize();
		cudaMemcpy(pHostOutImage, pDevOutImage, sizeof(unsigned char)*DIM*DIM, cudaMemcpyDeviceToHost);
	}
}

__global__ void RayTrace_Kernel(Sphere *Spheres, Plane *Planes, unsigned char *ptr, unsigned char *pDevGrayResult)
{
	int x0 = -DIM / 2;
	int y0 = DIM / 2;
	float z = -(DIM / 2) / tan(M_PI * 15 / 180.0);
	int x = TILE_WIDTH * blockIdx.x + threadIdx.x;	// 0 ~ 1023
	int y = TILE_WIDTH * blockIdx.y + threadIdx.y;	// 0 ~ 1023
	Vec3 p = { 80.0, 0.0, 0.0 };
	Vec3 v = { x0 + x - p[0], -y0 + y - p[1], z - p[2] };
	Normalize(v);
	Vec3 C = { 0.0, 0.0, 0.0 };
	 
	int sidx;
	int sidx2;
	int pidx;
	int pidx2;
	float t;
	float Pt;
	bool spherecheck = intersect_line_sphere(Spheres, p, v, sidx, t);
	bool planecheck = intersect_line_plane(Planes, p, v, pidx, Pt);

	if ((spherecheck && !planecheck) || (spherecheck && planecheck && (t <= Pt)))//구 먼저 교차
	{
		Vec3 pt;//교차점
		Line(pt, p, v, t);

		Vec3 N;// 교차점에서의 법선
		Sub(N, pt, Spheres[sidx].Pos);
		Normalize(N);

		Phong(Spheres, Planes, pt, N, Spheres[sidx], C, 1.0);

		// 교차점에서의 반사광선의 방향을 구한다.
		Vec3 R;
		R[0] = v[0] - 2.0 * Dot(N, v) * N[0];
		R[1] = v[1] - 2.0 * Dot(N, v) * N[1];
		R[2] = v[2] - 2.0 * Dot(N, v) * N[2];
		Normalize(R);

		spherecheck = intersect_line_sphere(Spheres, pt, R, sidx2, t);
		planecheck = intersect_line_plane(Planes, pt, R, pidx2, Pt);

		if ((spherecheck && !planecheck) || (spherecheck && planecheck && (t <= Pt)))//구 반사 후 구 교차
		{
			Vec3 pt2;
			Line(pt2, pt, R, t);// 교차점
			Vec3 N2;// 교차점에서의 법선
			Sub(N2, pt2, Spheres[sidx2].Pos);
			Normalize(N2);

			Phong(Spheres, Planes, pt2, N2, Spheres[sidx2], C, 0.3);
		}
		else if ((spherecheck && planecheck && (t > Pt)) || (!spherecheck && planecheck))//구 반사 후 평면 교차
		{
			Vec3 pt2;
			Line(pt2, pt, R, Pt);
			Vec3 N2;

			Vec3 temp1, temp2;
			Sub(temp1, Planes[pidx2].p2, Planes[pidx2].p1);
			Sub(temp2, Planes[pidx2].p3, Planes[pidx2].p1);
			Outer(N2, temp1, temp2);
			Normalize(N2);

			PhongP(Spheres, Planes, pt2, N2, Planes[pidx2], C, 0.3);
		}

		if (Spheres[sidx].bTransparent == true)//구가 투명할 때 굴절 구현
		{
			Vec3 pt4;
			float n = 1.0 / 1.015;//굴절률
			float cos1 = N[0] * (-v[0]) + N[1] * (-v[1]) + N[2] * (-v[2]);
			float cos2 = sqrt(1 - pow(n, 2) * (1 - pow(cos1, 2)));

			float a = cos2 - n * cos1;
			Vec3 t1 = { n*v[0] - a* N[0], n*v[1] - a* N[1], n*v[2] - a* N[2] };//굴절광선 방향
			Normalize(t1);

			Vec3 u;
			Sub(u, pt, Spheres[sidx].Pos);
			float r = Spheres[sidx].Rad;
			float D = SQR(Dot(u, t1)) - (Dot(u, u) - r * r);
			float t1t = -Dot(u, t1) + sqrt(D);//2번째 교차시간

			Vec3 N2;
			Line(pt4, pt, t1, t1t);//pt4는 2번째 교차점
			Sub(N2, Spheres[sidx].Pos, pt4);
			Normalize(N2);

			float cos3 = N2[0] * (-t1[0]) + N2[1] * (-t1[1]) + N2[2] * (-t1[2]);
			float cos4 = sqrt(1 - pow(1 / n, 2) * (1 - pow(cos3, 2)));

			float b = cos4 - 1 / n * cos3;
			Vec3 t2 = { 1 / n * t1[0] - b * N2[0], 1 / n * t1[1] - b * N2[1], 1 / n * t1[2] - b * N2[2] };
			Normalize(t2);

			spherecheck = intersect_line_sphere(Spheres, pt4, t2, sidx, t);
			planecheck = intersect_line_plane(Planes, pt4, t2, pidx, Pt);

			if ((spherecheck && !planecheck) || (spherecheck && planecheck && (t <= Pt)))//굴절광선이 구와 교차
			{
				Vec3 pt5;
				Line(pt5, pt4, t2, t);// 교차점
				Vec3 N2;// 교차점에서의 법선
				Sub(N2, pt5, Spheres[sidx].Pos);
				Normalize(N2);

				Phong(Spheres, Planes, pt5, N2, Spheres[sidx], C, 0.3);
			}
			else if ((spherecheck && planecheck && (t > Pt)) || (!spherecheck && planecheck))//굴절광선이 평면과 교차
			{
				Vec3 pt5;
				Line(pt5, pt4, t2, Pt);
				Vec3 N2;

				Vec3 temp1, temp2;
				Sub(temp1, Planes[pidx].p2, Planes[pidx].p1);
				Sub(temp2, Planes[pidx].p3, Planes[pidx].p1);
				Outer(N2, temp1, temp2);
				Normalize(N2);

				PhongP(Spheres, Planes, pt5, N2, Planes[pidx], C, 0.3);
			}
		}
	}
	else if ((spherecheck && planecheck && (t > Pt)) || (!spherecheck && planecheck))//평면 먼저 교차시
	{
		Vec3 pt;
		Line(pt, p, v, Pt);
		Vec3 N;

		Vec3 temp1, temp2;
		Sub(temp1, Planes[pidx].p2, Planes[pidx].p1);
		Sub(temp2, Planes[pidx].p3, Planes[pidx].p1);
		Outer(N, temp1, temp2);
		Normalize(N);

		PhongP(Spheres, Planes, pt, N, Planes[pidx], C, 1.0);

		Vec3 R;
		R[0] = v[0] - 2.0 * Dot(N, v) * N[0];
		R[1] = v[1] - 2.0 * Dot(N, v) * N[1];
		R[2] = v[2] - 2.0 * Dot(N, v) * N[2];
		Normalize(R);

		spherecheck = intersect_line_sphere(Spheres, pt, R, sidx2, t);
		planecheck = intersect_line_plane(Planes, pt, R, pidx2, Pt);

		if ((spherecheck && !planecheck) || (spherecheck && planecheck && (t <= Pt)))//평면 반사 후 구 교차
		{
			Vec3 pt2;
			Line(pt2, pt, R, t);// 교차점
			Vec3 N2;// 교차점에서의 법선
			Sub(N2, pt2, Spheres[sidx2].Pos);
			Normalize(N2);

			Phong(Spheres, Planes, pt2, N2, Spheres[sidx2], C, 0.3);
		}
		else if ((spherecheck && planecheck && (t > Pt)) || (!spherecheck && planecheck))//평면 반사 후 평면 교차
		{
			Vec3 pt2;
			Line(pt2, pt, R, Pt);
			Vec3 N2;

			Vec3 temp1, temp2;
			Sub(temp1, Planes[pidx2].p2, Planes[pidx2].p1);
			Sub(temp2, Planes[pidx2].p3, Planes[pidx2].p1);
			Outer(N2, temp1, temp2);
			Normalize(N2);

			PhongP(Spheres, Planes, pt2, N2, Planes[pidx2], C, 0.3);
		}
	}
	int offset = (y * DIM + x) * 3;
	ptr[offset] = (C[0] > 1.0) ? 255 : (unsigned int)(C[0] * 255);
	ptr[offset + 1] = (C[1] > 1.0) ? 255 : (unsigned int)(C[1] * 255);
	ptr[offset + 2] = (C[2] > 1.0) ? 255 : (unsigned int)(C[2] * 255);
	pDevGrayResult[y * DIM + x] = (C[0] > 1.0) ? 255 : (unsigned int)(C[0] * 255);
}

/*
brief 장면에서 광선과 교차하는 구의 인덱스(sidx)와 교차하는 시간(t)를 찾는다.
param ray 광선의 방정식
param sidx 광선과 교차하는 구의 인덱스가 저장됨
param t 광선이 구와 교차하는 시간 (파라미터)가 저장됨
return 광선이 구와 교차하면 true, 아니면 false를 반환한다.
*/
__device__ bool intersect_line_sphere(Sphere *Spheres, Vec3 p, Vec3 v, int &sidx, float &t)
{
	sidx = -1;
	float tempt;
	t = 1.2e+15;

	for (int i = 0; i < SPHERES; ++i)
	{
		Vec3 u;
		Sub(u, p, Spheres[i].Pos);
		float r = Spheres[i].Rad;
		float D = (Dot(u, v) * Dot(u, v)) - (Dot(u, u) - r * r);
		if (D >= 0.0)
		{
			tempt = -Dot(u, v) - sqrt(D);
			if (t > tempt && tempt > 0.0)
			{
				t = tempt;
				sidx = i;
			}
		}
	}
	if (sidx != -1)
		return true;
	return false;
}

__device__ bool intersect_line_plane(Plane *Planes, Vec3 p, Vec3 v, int &pidx, float &t)
{
	pidx = -1;
	float temppt;
	t = 1.2e+15;
	Vec3 P;
	float xmin, xmax;
	float ymin, ymax;
	float zmin, zmax;

	for (int i = 0; i < PLANES; ++i)
	{
		Vec3 u;
		Sub(u, p, Planes[i].p1);
		Vec3 n;
		Vec3 temp1, temp2;
		Sub(temp1, Planes[i].p2, Planes[i].p1);
		Sub(temp2, Planes[i].p3, Planes[i].p1);
		Outer(n, temp1, temp2);
		Normalize(n);
		if (ABS(Dot(v, n)) < 0.0)
			temppt = 0.0;
		else
			temppt = -Dot(u, n) / Dot(v, n);

		Line(P, p, v, temppt);

		xmin = (Planes[i].p1[0] >= Planes[i].p3[0] ? Planes[i].p3[0] : Planes[i].p1[0]);
		xmax = (Planes[i].p3[0] <= Planes[i].p1[0] ? Planes[i].p1[0] : Planes[i].p3[0]);
		ymin = (Planes[i].p1[1] >= Planes[i].p3[1] ? Planes[i].p3[1] : Planes[i].p1[1]);
		ymax = (Planes[i].p3[1] <= Planes[i].p1[1] ? Planes[i].p1[1] : Planes[i].p3[1]);
		zmin = (Planes[i].p1[2] >= Planes[i].p3[2] ? Planes[i].p3[2] : Planes[i].p1[2]);
		zmax = (Planes[i].p3[2] <= Planes[i].p1[2] ? Planes[i].p1[2] : Planes[i].p3[2]);

		if ((P[0] >= xmin - 0.00001) && (P[0] <= xmax + 0.00001) && (P[1] >= ymin - 0.00001)
			&& (P[1] <= ymax + 0.00001) && (P[2] >= zmin - 0.00001) && (P[2] <= zmax + 0.00001))
		{
			if ((t > temppt) && (temppt > 0.0 + 0.01))
			{
				t = temppt;
				pidx = i;
			}
		}
	}
	if (pidx != -1)
		return true;
	return false;
}

/*
brief 조명 모델을 통해 구 교차점에서의 색상을 계산한다.
param P 교차점의 좌표
param N 교차점의 법선
param Obj 교차하는 구(재질의 정보 포함)
return 계산된 색상을 반환한다.
*/
__device__ void Phong(Sphere *Spheres, Plane *Planes, Vec3 P, Vec3 N, Sphere S, Vec3 C, float k)
{
	Vec3 LightPos = { -1000.0, 1500.0, 700.0 };
	Vec3 L;
	Sub(L, LightPos, P);
	Normalize(L);

	Vec3 V;
	V[0] = -P[0];
	V[1] = -P[1];
	V[2] = -P[2];
	Normalize(V);

	float NdotL = Dot(N, L);
	Vec3 R;
	R[0] = -L[0] + 2.0 * NdotL * N[0];
	R[1] = -L[1] + 2.0 * NdotL * N[1];
	R[2] = -L[2] + 2.0 * NdotL * N[2];
	Normalize(R);

	Vec3 Ambi, Diff, Spec;
	Ambi[0] = 0.2 * S.Ka[0];
	Ambi[1] = 0.2 * S.Ka[1];
	Ambi[2] = 0.2 * S.Ka[2];
	Diff[0] = 1.0 * S.Kd[0];
	Diff[1] = 1.0 * S.Kd[1];
	Diff[2] = 1.0 * S.Kd[2];
	Spec[0] = 1.0 * S.Ks[0];
	Spec[1] = 1.0 * S.Ks[1];
	Spec[2] = 1.0 * S.Ks[2];

	float VdotR = Dot(V, R);

	int sidx;//광선과 교차하는 가장 가까운 구의 인덱스
	int pidx;//광선과 교차하는 가장 가까운 평면의 인덱스
	float t;//구와 교차하는 시간
	float pt;//평면과 교차하는 시간

	if ((intersect_line_sphere(Spheres, P, L, sidx, t) && (t > 0.0 + 0.01)) || (intersect_line_plane(Planes, P, L, pidx, pt) && (pt > 0.0 + 0.01)))
	{
		C[0] += k * Ambi[0];
		C[1] += k * Ambi[1];
		C[2] += k * Ambi[2];
	}
	else
	{
		C[0] += k * (Ambi[0] + Diff[0] * MAX(0.0, NdotL) + Spec[0] * powf(MAX(0.0, VdotR), S.ns));
		C[1] += k * (Ambi[1] + Diff[1] * MAX(0.0, NdotL) + Spec[1] * powf(MAX(0.0, VdotR), S.ns));
		C[2] += k * (Ambi[2] + Diff[2] * MAX(0.0, NdotL) + Spec[2] * powf(MAX(0.0, VdotR), S.ns));
	}
}
//조명 모델을 통해 평면 교차점에서의 색상을 계산한다.
__device__ void PhongP(Sphere *Spheres, Plane *Planes, Vec3 P, Vec3 N, Plane PL, Vec3 C, float k)
{
	Vec3 LightPos = { -1000.0, 1500.0, 700.0 };
	Vec3 L;
	Sub(L, LightPos, P);
	Normalize(L);

	Vec3 V;
	V[0] = -P[0];
	V[1] = -P[1];
	V[2] = -P[2];
	Normalize(V);

	float NdotL = Dot(N, L);
	Vec3 R;
	R[0] = -L[0] + 2.0 * NdotL * N[0];
	R[1] = -L[1] + 2.0 * NdotL * N[1];
	R[2] = -L[2] + 2.0 * NdotL * N[2];
	Normalize(R);

	Vec3 Ambi, Diff, Spec;
	Ambi[0] = 0.2 * PL.Ka[0];
	Ambi[1] = 0.2 * PL.Ka[1];
	Ambi[2] = 0.2 * PL.Ka[2];
	Diff[0] = 1.0 * PL.Kd[0];
	Diff[1] = 1.0 * PL.Kd[1];
	Diff[2] = 1.0 * PL.Kd[2];
	Spec[0] = 1.0 * PL.Ks[0];
	Spec[1] = 1.0 * PL.Ks[1];
	Spec[2] = 1.0 * PL.Ks[2];

	float VdotR = Dot(V, R);

	int sidx;//광선과 교차하는 가장 가까운 구의 인덱스
	int pidx;//광선과 교차하는 가장 가까운 평면의 인덱스
	float t;//구와 교차하는 시간
	float pt;//평면과 교차하는 시간

	if ((intersect_line_sphere(Spheres, P, L, sidx, t) && (t > 0.0 + 0.000001)) || (intersect_line_plane(Planes, P, L, pidx, pt) && (pt > 0.0 + 0.0001)))
	{
		C[0] += k * Ambi[0];
		C[1] += k * Ambi[1];
		C[2] += k * Ambi[2];
	}
	else
	{
		C[0] += k * (Ambi[0] + Diff[0] * MAX(0.0, NdotL) + Spec[0] * powf(MAX(0.0, VdotR), PL.ns));
		C[1] += k * (Ambi[1] + Diff[1] * MAX(0.0, NdotL) + Spec[1] * powf(MAX(0.0, VdotR), PL.ns));
		C[2] += k * (Ambi[2] + Diff[2] * MAX(0.0, NdotL) + Spec[2] * powf(MAX(0.0, VdotR), PL.ns));
	}
}


void Close()
{
	printf("Close callback invoked...\n");
	cudaFree(pDevSrcImage);
	cudaFree(pDevOutImage);
	cudaFree(pDevSobelResult);
	cudaFree(pDevGrayResult);
	cudaFree(SphereList);
	cudaFree(PlaneList);
	cudaDeviceReset();
}

__device__ void Sub(Vec3 c, Vec3 a, Vec3 b)
{
	c[0] = a[0] - b[0];
	c[1] = a[1] - b[1];
	c[2] = a[2] - b[2];
}

__device__ float Dot(Vec3 a, Vec3 b)
{
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ void Normalize(Vec3 a)
{
	float len = sqrtf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
	a[0] /= len;
	a[1] /= len;
	a[2] /= len;
}

__device__ void Line(Vec3 pt, Vec3 pos, Vec3 dir, float time)
{
	pt[0] = pos[0] + dir[0] * time;
	pt[1] = pos[1] + dir[1] * time;
	pt[2] = pos[2] + dir[2] * time;
}

__device__ void Outer(Vec3 c, Vec3 a, Vec3 b)
{
	c[0] = a[1] * b[2] - a[2] * b[1];
	c[1] = a[2] * b[0] - a[0] * b[2];
	c[2] = a[0] * b[1] - a[1] * b[0];
}

__global__ void SobelEdge(unsigned char *pDevSrc, float *pDevSobelResult, int S)
{
	int MaskSobelX[9] = { -1, 0, 1 ,-2, 0, 2 ,-1, 0, 1 };
	int MaskSobelY[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

	int i = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int j = blockIdx.x * TILE_WIDTH + threadIdx.x;

	int Gx = 0;
	int Gy = 0;

	if (i > 0 && j > 0 && i < S - 1 && j < S - 1)
	{
		for (int r = 0; r < 3; r++)
		{
			for (int c = 0; c < 3; c++)
			{
				int idx = (i - 1 + r)*S + j - 1 + c;

				Gx += (MaskSobelX[r * 3 + c] * pDevSrc[idx]);
				Gy += (MaskSobelY[r * 3 + c] * pDevSrc[idx]);
			}
		}
		pDevSobelResult[i*S + j] = sqrtf(Gx*Gx + Gy * Gy);
	}
}

__global__ void SetEdgeColor(unsigned char *pDevOut, float *pDevSobelResult, int min, int max)
{
	int i = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int j = blockIdx.x * TILE_WIDTH + threadIdx.x;

	if (i > 0 && j > 0 && i < DIM - 1 && j < DIM - 1)
	{
		float t = (pDevSobelResult[i * DIM + j] - min) / (max - min);
		pDevOut[i * DIM + j] = (unsigned char)(255 * t);
	}
}

__global__ void Gaussian(unsigned char *pDevSrc, unsigned char *pDevOut, int S)
{
	int MaskGaussian[3][3] = {
		{ 1,2,1 },
		{ 2,4,2 },
		{ 1,2,1 }
	};

	int i = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int j = blockIdx.x * TILE_WIDTH + threadIdx.x;
	double GaussianResult = 0;

	if (i > 0 && j > 0 && i < S - 1 && j < S - 1)
	{
		for (int r = 0; r < 3; r++)//row
		{
			for (int c = 0; c < 3; c++)//column
			{
				int idx = (i - 1 + r) * S + j - 1 + c;
				GaussianResult += (MaskGaussian[r][c] * pDevSrc[idx]);
			}
		}
		GaussianResult /= 20;
		pDevOut[i*S + j] = (unsigned char)GaussianResult;
	}
}

__global__ void Median(unsigned char *pDevSrc, unsigned char *pDevOut, unsigned char *pTempImage, int S)
{
	const int MaskSize = 9;
	int hold;
	int k;
	k = MaskSize / 2;

	int i = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int j = blockIdx.x * TILE_WIDTH + threadIdx.x;

	int Num[MaskSize];

	for (int n = 0; n < MaskSize; n++)
		Num[n] = pDevSrc[i*S + j + (n - k)];
	

	for (int loop = 0; loop < MaskSize; loop++)
	{
		for (int m = 0; m < MaskSize - 1; m++)
		{
			if (Num[m] > Num[m + 1])
			{
				hold = Num[m];
				Num[m] = Num[m + 1];
				Num[m + 1] = hold;
			}
		}
	}
	pTempImage[i*S + j] = Num[k];
}

__global__ void SetMedianResult(unsigned char *pDevOut, unsigned char *pTempImage)
{
	int i = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int j = blockIdx.x * TILE_WIDTH + threadIdx.x;

	if (i > 0 && j > 0 && i < DIM - 1 && j < DIM - 1)
	{
		if (pTempImage[i*DIM + j] < 0)
			pDevOut[i*DIM + j] = 0;
		else if (pTempImage[i*DIM + j] > 255)
			pDevOut[i*DIM + j] = 255;
		else
			pDevOut[i*DIM + j] = pTempImage[i*DIM + j];
	}
}

void Keyboard(unsigned char key, int x, int y)
{
	if (key == 27)
		exit(-1);
	else if (key == 'o')
		keynum = 0;
	else if (key == 'h')
		keynum = 1;
	else if (key == 's')
		keynum = 2;
	else if (key == 'g')
		keynum = 3;
	else if (key == 'm')
		keynum = 4;

	glutPostRedisplay();
}