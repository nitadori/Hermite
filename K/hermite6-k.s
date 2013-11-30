	.ident	"$Options: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) --preinclude //opt/FJSVfxlang/1.2.1/bin/../lib/FCC.pre --g++ -D__FUJITSU -Dunix -Dsparc -D__sparc__ -D__unix -D__sparc -D__BUILTIN_VA_ARG_INCR -D_OPENMP=200805 -D__PRAGMA_REDEFINE_EXTNAME -D__FCC_VERSION=600 -D__USER_LABEL_PREFIX__= -D__OPTIMIZE__ -D__HPC_ACE__ -D__ELF__ -D__linux -Asystem(unix) -Dlinux -D__LIBC_6B -D_LP64 -D__LP64__ --K=omp -DSIXTH -DHPC_ACE_GRAVITY -I/opt/FJSVfxlang/1.2.1/include/mpi/fujitsu --K=noocl -D_REENTRANT -D__MT__ --lp --zmode=64 --sys_include=//opt/FJSVfxlang/1.2.1/bin/../include/c++/std --sys_include=//opt/FJSVfxlang/1.2.1/bin/../include/c++ --sys_include=//opt/FJSVfxlang/1.2.1/bin/../include --sys_include=/opt/FJSVXosDevkit/sparc64fx/target/usr/include --K=opt -D__sparcv9 -D__sparc_v9__ -D__arch64__ --exceptions ../SRC/hermite6-k.cpp -- -ncmdname=FCCpx -Nnoline -Kdalign -zobe=no-static-clump -zobe=cplus -O3 -x- -KSPARC64IXfx,dalign,ns,mfunc,lib,eval,rdconv,prefetch_conditional,fp_contract,fp_relaxed,ilfunc,fast_matmul -Kthreadsafe -O3 -x- -KSPARC64IXfx,dalign,ns,mfunc,lib,eval,rdconv,prefetch_conditional,fp_contract,fp_relaxed,ilfunc,fast_matmul,uxsimd,optmsg=2 -x32 -Nsrc -Kopenmp,threadsafe -KLP -zsrc=../SRC/hermite6-k.cpp hermite6-k.s $"
	.file	"hermite6-k.cpp"
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite6-k.cpp _ZNKSt12_String_baseIcSt20__iostring_allocatorIcEE21_M_throw_length_errorEv $"
	.section	".text._ZNKSt12_String_baseIcSt20__iostring_allocatorIcEE21_M_throw_length_errorEv",#alloc,#execinstr

	.weak	_ZNKSt12_String_baseIcSt20__iostring_allocatorIcEE21_M_throw_length_errorEv
	.align	64
_ZNKSt12_String_baseIcSt20__iostring_allocatorIcEE21_M_throw_length_errorEv:
.LLFB1:
.L205:

/*    635 */	save	%sp,-192,%sp
.LLCFI0:


.L206:

/*    636 */	sethi	%h44(_ZZNKSt12_String_baseIcSt20__iostring_allocatorIcEE21_M_throw_length_errorEvEs),%o0

/*    636 */	or	%o0,%m44(_ZZNKSt12_String_baseIcSt20__iostring_allocatorIcEE21_M_throw_length_errorEvEs),%o0

/*    636 */	sllx	%o0,12,%o0


/*    636 */	call	_ZSt24__stl_throw_length_errorPKc
/*    636 */	or	%o0,%l44(_ZZNKSt12_String_baseIcSt20__iostring_allocatorIcEE21_M_throw_length_errorEvEs),%o0


.L207:

/*    637 */	ret
	restore



.LLFE1:
	.size	_ZNKSt12_String_baseIcSt20__iostring_allocatorIcEE21_M_throw_length_errorEv,.-_ZNKSt12_String_baseIcSt20__iostring_allocatorIcEE21_M_throw_length_errorEv
	.type	_ZNKSt12_String_baseIcSt20__iostring_allocatorIcEE21_M_throw_length_errorEv,#function
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite6-k.cpp _ZNSt12_String_baseIcSt20__iostring_allocatorIcEE17_M_allocate_blockEm $"
	.section	".text._ZNSt12_String_baseIcSt20__iostring_allocatorIcEE17_M_allocate_blockEm",#alloc,#execinstr

	.weak	_ZNSt12_String_baseIcSt20__iostring_allocatorIcEE17_M_allocate_blockEm
	.align	64
_ZNSt12_String_baseIcSt20__iostring_allocatorIcEE17_M_allocate_blockEm:
.LLFB2:
.L208:

/*    645 */	save	%sp,-192,%sp
.LLCFI1:


.L209:

/*    652 */	cmp	%i1,-1

/*    652 */	bgu,pn	%xcc, .L211
	nop


.L212:

/*    652 */	cmp	%i1,%g0

/*    652 */	bne,pt	%xcc, .L210
	nop


.L211:

/*    636 */	sethi	%h44(_ZZNKSt12_String_baseIcSt20__iostring_allocatorIcEE21_M_throw_length_errorEvEs),%o0

/*    636 */	or	%o0,%m44(_ZZNKSt12_String_baseIcSt20__iostring_allocatorIcEE21_M_throw_length_errorEvEs),%o0

/*    636 */	sllx	%o0,12,%o0


/*    636 */	call	_ZSt24__stl_throw_length_errorPKc
/*    636 */	or	%o0,%l44(_ZZNKSt12_String_baseIcSt20__iostring_allocatorIcEE21_M_throw_length_errorEvEs),%o0


.L1106:

/*    637 */	ret
	restore



.L210:


/*     61 */	cmp	%i1,257

/*     61 */	bleu,pt	%xcc, .L1103
/*     61 */	add	%i0,16,%o0


.L1102:

/*    458 */	cmp	%i1,-1

/*    458 */	bgu,pn	%xcc, .L1075
	nop


.L1083:


/*    123 */	call	_Znwm
/*    123 */	mov	%i1,%o0


.L5525:

/*    123 */	cmp	%o0,%g0

/*    123 */	be,pt	%xcc, .L1086
	nop


.L1090:


.L1103:

/*    660 */	add	%o0,%i1,%i1

/*    657 */	stx	%o0,[%i0]

/*    658 */	stx	%o0,[%i0+8]

/*    660 */	stx	%i1,[%i0+280]

/*      0 */	ret
	restore



.L1086:


/*    123 */	call	__cxa_allocate_exception
/*    123 */	mov	8,%o0
/*    123 */	mov	%o0,%l1
/*    123 */	call	_ZNSt9bad_allocC1Ev
/*    123 */	mov	%l1,%o0


.L5939:

/*    123 */	sethi	%h44(_ZTISt9bad_alloc),%o1

/*    123 */	sethi	%h44(_ZNSt9bad_allocD1Ev),%o2

/*    123 */	or	%o1,%m44(_ZTISt9bad_alloc),%o1

/*    123 */	or	%o2,%m44(_ZNSt9bad_allocD1Ev),%o2

/*    123 */	sllx	%o1,12,%o1

/*    123 */	sllx	%o2,12,%o2

/*    123 */	or	%o1,%l44(_ZTISt9bad_alloc),%o1

/*    123 */	or	%o2,%l44(_ZNSt9bad_allocD1Ev),%o2


/*    123 */	call	__cxa_throw
/*    123 */	mov	%l1,%o0


.L1075:


/*    459 */	call	__cxa_allocate_exception
/*    459 */	mov	8,%o0
/*    459 */	mov	%o0,%l0
/*    459 */	call	_ZNSt9bad_allocC1Ev
/*    459 */	mov	%l0,%o0


.L5938:

/*    459 */	sethi	%h44(_ZTISt9bad_alloc),%o1

/*    459 */	sethi	%h44(_ZNSt9bad_allocD1Ev),%o2

/*    459 */	or	%o1,%m44(_ZTISt9bad_alloc),%o1

/*    459 */	or	%o2,%m44(_ZNSt9bad_allocD1Ev),%o2

/*    459 */	sllx	%o1,12,%o1

/*    459 */	sllx	%o2,12,%o2

/*    459 */	or	%o1,%l44(_ZTISt9bad_alloc),%o1

/*    459 */	or	%o2,%l44(_ZNSt9bad_allocD1Ev),%o2


/*    459 */	call	__cxa_throw
/*    459 */	mov	%l0,%o0


.L214:


.LLFE2:
	.size	_ZNSt12_String_baseIcSt20__iostring_allocatorIcEE17_M_allocate_blockEm,.-_ZNSt12_String_baseIcSt20__iostring_allocatorIcEE17_M_allocate_blockEm
	.type	_ZNSt12_String_baseIcSt20__iostring_allocatorIcEE17_M_allocate_blockEm,#function
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite6-k.cpp _ZN7Gravity6GForceC1Ev $"
	.section	".text._ZN7Gravity6GForceC1Ev",#alloc,#execinstr

	.weak	_ZN7Gravity6GForceC1Ev
	.align	64
_ZN7Gravity6GForceC1Ev:
.LLFB3:
.L526:

/*     34 */

.L527:


/*     25 */	sxar2
/*     25 */	fzero,s	%f32
/*     25 */	std,s	%f32,[%o0]


/*     25 */	sxar2
/*     25 */	std,s	%f32,[%o0+16]
/*     25 */	std,s	%f32,[%o0+32]


/*     25 */	sxar2
/*     25 */	std,s	%f32,[%o0+48]
/*     25 */	std,s	%f32,[%o0+64]


/*     25 */	sxar2
/*     25 */	std,s	%f32,[%o0+80]
/*     25 */	std,s	%f32,[%o0+96]


/*     25 */	sxar2
/*     25 */	std,s	%f32,[%o0+112]
/*     25 */	std,s	%f32,[%o0+128]

/*     25 */	retl
	nop



.L528:


.LLFE3:
	.size	_ZN7Gravity6GForceC1Ev,.-_ZN7Gravity6GForceC1Ev
	.type	_ZN7Gravity6GForceC1Ev,#function
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite6-k.cpp _ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE $"
	.section	".text"
	.global	_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE
	.align	64
_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE:
.LLFB4:
.L538:

/*      6 */	save	%sp,-304,%sp
.LLCFI2:
/*      6 */	stx	%i2,[%fp+2191]
/*      6 */	stx	%i3,[%fp+2199]

.L539:

/*     26 */	sxar1
/*     26 */	fmovd	%f2,%f258


/*     13 */	srl	%i0,31,%g1

/*     13 */	add	%g1,%i0,%g1

/*     13 */	sra	%g1,1,%g1

/*     13 */	stw	%g1,[%fp+2031]

/*     26 */	sxar1
/*     26 */	std,s	%f2,[%fp+1935]

/*     14 *//*     14 */	sethi	%h44(_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE._OMP_1),%o0
/*     14 */	mov	%fp,%o1
/*     14 */	or	%o0,%m44(_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE._OMP_1),%o0
/*     14 */	mov	%g0,%o2
/*     14 */	sllx	%o0,12,%o0
/*     14 */	call	__mpc_opar
/*     14 */	or	%o0,%l44(_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE._OMP_1),%o0

/*     44 */
/*     44 */	ret
	restore



.L565:


.LLFE4:
	.size	_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE,.-_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE
	.type	_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE,#function
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite6-k.cpp _ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE._OMP_1 $"
	.section	".text"
	.align	64
_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE._OMP_1:
.LLFB5:
.L5710:

/*     14 */	save	%sp,-768,%sp
.LLCFI3:
/*     14 */	stx	%i0,[%fp+2175]
/*     14 */	stx	%i3,[%fp+2199]
/*     14 */	stx	%i0,[%fp+2175]

.L5711:

/*     14 *//*     14 */	sxar1
/*     14 */	ldsw	[%i0+2035],%xg13
/*     14 */
/*     14 */
/*     14 */
/*     15 */	ldsw	[%i0+2031],%l0
/*     15 */	cmp	%l0,%g0
/*     15 */	ble	.L5726
/*     15 */	mov	%g0,%o0


.L5712:

/*     15 */	sethi	%h44(.LR0.cnt.5),%g1

/*     15 */	sethi	%h44(.LR0.cnt.1),%g2

/*     15 */	or	%g1,%m44(.LR0.cnt.5),%g1

/*     15 */	or	%g2,%m44(.LR0.cnt.1),%g2

/*     15 */	sllx	%g1,12,%g1

/*     15 */	sllx	%g2,12,%g2

/*     15 */	or	%g1,%l44(.LR0.cnt.5),%g1

/*     15 */	or	%g2,%l44(.LR0.cnt.1),%g2

/*     15 */	sethi	%h44(.LR0.cnt.6),%g3


/*     15 */	sxar2
/*     15 */	ldd	[%g1],%f220
/*     15 */	sethi	%h44(.LR0.cnt.2),%xg0

/*     15 */	sxar1
/*     15 */	ldd	[%g1],%f476


/*     15 */	or	%g3,%m44(.LR0.cnt.6),%g3


/*     15 */	sxar2
/*     15 */	or	%xg0,%m44(.LR0.cnt.2),%xg0
/*     15 */	ldd	[%g2],%f222

/*     15 */	sxar1
/*     15 */	ldd	[%g2],%f478

/*     15 */	sllx	%g3,12,%g3

/*     15 */	sxar1
/*     15 */	sllx	%xg0,12,%xg0

/*     15 */	or	%g3,%l44(.LR0.cnt.6),%g3


/*     15 */	sxar2
/*     15 */	or	%xg0,%l44(.LR0.cnt.2),%xg0
/*     15 */	mov	1,%xg12

/*     15 */	sra	%l0,%g0,%l0


/*     15 */	sxar2
/*     15 */	ldd	[%g3],%f224
/*     15 */	ldd	[%g3],%f480




/*     15 */	sxar2
/*    ??? */	std,s	%f220,[%fp+1519]
/*     15 */	ldd	[%xg0],%f226



/*     15 */	sxar2
/*     15 */	ldd	[%xg0],%f482
/*     15 */	stx	%xg12,[%fp+2031]


/*     15 */	sxar2
/*    ??? */	std,s	%f222,[%fp+1503]
/*    ??? */	std,s	%f224,[%fp+1551]

/*     15 */	sxar1
/*    ??? */	std,s	%f226,[%fp+1535]

.L5713:

/*     15 */	add	%fp,2039,%l1

/*     15 */	mov	1,%l5

/*     15 */	add	%fp,2023,%l2

/*     15 */	add	%fp,2031,%l3

/*     15 */	sra	%l5,%g0,%l4

.L5714:

/*     15 */	sra	%o0,%g0,%o0

/*     15 */	stx	%g0,[%sp+2223]

/*     15 */	mov	1,%o2

/*     15 */	mov	%g0,%o3

/*     15 */	mov	%l0,%o1

/*     15 */	mov	%l1,%o4


/*     15 */	stx	%g0,[%sp+2231]

/*     15 */	stx	%l3,[%sp+2239]


/*     15 */	sxar2
/*     15 */	ldx	[%fp+2199],%xg10
/*     15 */	stx	%xg10,[%sp+2247]

/*     15 */	call	__mpc_ostd_th
/*     15 */	mov	%l2,%o5
/*     15 */	sxar2
/*     15 */	ldx	[%fp+2031],%xg11
/*     15 */	cmp	%xg11,%g0
/*     15 */	ble,pn	%xcc, .L5726
	nop


.L5715:

/*     15 */	ldx	[%fp+2039],%o0


/*     15 */	sxar2
/*     15 */	ldx	[%fp+2023],%xg0
/*     15 */	ldx	[%i0+2191],%xg5


/*     15 */	sxar2
/*     15 */	ldx	[%i0+2199],%xg6
/*     15 */	ldd,s	[%i0+1935],%f34

/*     15 */	sra	%o0,%g0,%o0


/*     15 */	sxar2
/*     15 */	sra	%xg0,%g0,%xg0
/*     15 */	sub	%xg0,%o0,%xg0


/*     15 */	sxar2
/*     15 */	sra	%o0,%g0,%xg1
/*     15 */	add	%xg0,1,%xg0


/*     15 */	sxar2
/*     15 */	sllx	%xg1,2,%xg2
/*     15 */	sra	%xg0,%g0,%xg0


/*     15 */	sxar2
/*     15 */	add	%xg2,%xg1,%xg2
/*     15 */	sub	%l4,%xg0,%xg0


/*     15 */	sxar2
/*     15 */	sllx	%xg2,6,%xg3
/*     15 */	srax	%xg0,32,%xg4


/*     15 */	sxar2
/*     15 */	sllx	%xg2,5,%xg2
/*     15 */	and	%xg0,%xg4,%xg0


/*     15 */	sxar2
/*     15 */	add	%xg5,%xg3,%xg5
/*     15 */	sub	%l5,%xg0,%xg0


/*     15 */	sxar2
/*     15 */	add	%xg6,%xg2,%xg6
/*     15 */	cmp	%xg0,8

/*     15 */	bl	.L5890
	nop


.L5886:


.L5894:


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg5+112],%f32
/*     15 */	add	%xg5,320,%xg7


/*     15 */	sxar2
/*     15 */	add	%xg5,640,%xg8
/*    ??? */	ldd,s	[%fp+1535],%f204


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg5+432],%f36
/*     15 */	ldd,s	[%xg5+272],%f60


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg5+224],%f58
/*    ??? */	ldd,s	[%fp+1551],%f230


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg5+176],%f84
/*     15 */	ldd,s	[%xg5+288],%f66


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg5+240],%f64
/*     15 */	fsubd,s	%f34,%f32,%f32


/*     15 */	sxar2
/*     15 */	fsubd,s	%f34,%f36,%f36
/*    ??? */	ldd,s	[%fp+1503],%f252


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg5+592],%f80
/*     15 */	ldd,s	[%xg5+128],%f92


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg5+304],%f72
/*     15 */	ldd,s	[%xg5+256],%f70


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg5+544],%f76
/*     15 */	ldd,s	[%xg5+192],%f86


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg5+752],%f54
/*    ??? */	ldd,s	[%fp+1519],%f226


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg5+64],%f96
/*     15 */	fmuld,s	%f204,%f32,%f38


/*     15 */	sxar2
/*     15 */	fmuld,s	%f230,%f32,%f40
/*     15 */	ldd,s	[%xg5+208],%f88


/*     15 */	sxar2
/*     15 */	fmuld,s	%f252,%f32,%f42
/*     15 */	fmuld,s	%f204,%f36,%f44


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg5+496],%f90
/*     15 */	fmuld,s	%f230,%f36,%f48


/*     15 */	sxar2
/*     15 */	fmuld,s	%f252,%f36,%f52
/*     15 */	ldd,s	[%xg5+144],%f94


/*     15 */	sxar2
/*     15 */	fmuld,s	%f226,%f32,%f46
/*     15 */	fmuld,s	%f226,%f36,%f50


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg5],%f176
/*     15 */	fsubd,s	%f34,%f54,%f54


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg5+80],%f110
/*     15 */	ldd,s	[%xg5+160],%f106


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f38,%f60,%f58,%f56
/*     15 */	fmaddd,s	%f38,%f66,%f64,%f62


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f40,%f60,%f58,%f68
/*     15 */	fmaddd,s	%f38,%f72,%f70,%f38


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f40,%f66,%f64,%f74
/*     15 */	fmaddd,s	%f42,%f60,%f58,%f60


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f44,%f80,%f76,%f78
/*     15 */	fmaddd,s	%f40,%f72,%f70,%f82


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f42,%f66,%f64,%f66
/*     15 */	fmaddd,s	%f40,%f56,%f84,%f56


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f40,%f62,%f86,%f62
/*     15 */	fmaddd,s	%f42,%f68,%f84,%f68


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f40,%f38,%f88,%f40
/*     15 */	fmaddd,s	%f42,%f74,%f86,%f74


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f46,%f60,%f84,%f60
/*     15 */	fmaddd,s	%f48,%f78,%f90,%f78


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f42,%f56,%f92,%f56
/*     15 */	fmaddd,s	%f42,%f62,%f94,%f62


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f46,%f68,%f92,%f68
/*     15 */	fmaddd,s	%f46,%f56,%f96,%f56

.L5716:


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+288],%f210
/*     15 */	fmaddd,s	%f32,%f56,%f176,%f56


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+240],%f206
/*     15 */	fmaddd,s	%f42,%f40,%f106,%f40


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f46,%f62,%f110,%f62
/*     15 */	ldd,s	[%xg5+96],%f214


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f42,%f82,%f88,%f82
/*     15 */	ldd,s	[%xg5+16],%f216


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f46,%f74,%f94,%f74
/*     15 */	fmaddd,s	%f42,%f72,%f70,%f42


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+432],%f254
/*     15 */	fmaddd,s	%f32,%f68,%f96,%f68


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f46,%f66,%f86,%f66
/*     15 */	ldd,s	[%xg8+272],%f242


/*     15 */	sxar2
/*     15 */	fmuld,s	%f204,%f54,%f202
/*     15 */	ldd,s	[%xg7+128],%f218


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f32,%f60,%f92,%f60
/*     15 */	ldd,s	[%xg7+304],%f222


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f44,%f210,%f206,%f208
/*     15 */	ldd,s	[%xg7+256],%f220


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+224],%f238
/*     15 */	fmaddd,s	%f48,%f80,%f76,%f212


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+192],%f232
/*     15 */	fmaddd,s	%f46,%f40,%f214,%f40


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+64],%f244
/*     15 */	fmaddd,s	%f32,%f62,%f216,%f62


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg5+32],%f236
/*     15 */	fmaddd,s	%f46,%f82,%f106,%f82


/*     15 */	sxar2
/*     15 */	std,s	%f56,[%xg6]
/*     15 */	fmaddd,s	%f32,%f74,%f110,%f74


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f46,%f42,%f88,%f46
/*     15 */	std,s	%f68,[%xg6+64]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f52,%f78,%f218,%f78
/*     15 */	fmaddd,s	%f32,%f66,%f94,%f66


/*     15 */	sxar2
/*     15 */	std,s	%f60,[%xg6+112]
/*     15 */	fmaddd,s	%f44,%f222,%f220,%f44


/*     15 */	sxar2
/*     15 */	fmuld,s	%f226,%f54,%f224
/*     15 */	fmuld,s	%f230,%f54,%f228


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f48,%f208,%f232,%f208
/*     15 */	ldd,s	[%xg7+208],%f246


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+176],%f38
/*     15 */	ldd,s	[%xg7+144],%f248


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f48,%f210,%f206,%f234
/*     15 */	fmaddd,s	%f32,%f40,%f236,%f40


/*     15 */	sxar2
/*     15 */	std,s	%f62,[%xg6+16]
/*     15 */	fmaddd,s	%f52,%f212,%f90,%f212


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f32,%f82,%f214,%f82
/*     15 */	std,s	%f74,[%xg6+80]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f52,%f80,%f76,%f80
/*     15 */	fmaddd,s	%f32,%f46,%f106,%f32


/*     15 */	sxar2
/*     15 */	std,s	%f66,[%xg6+128]
/*     15 */	fmaddd,s	%f202,%f242,%f238,%f240


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f50,%f78,%f244,%f78
/*     15 */	ldd,s	[%xg7],%f42


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f48,%f44,%f246,%f44
/*     15 */	ldd,s	[%xg5+48],%f58


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+80],%f56
/*     15 */	fmaddd,s	%f52,%f208,%f248,%f208


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+160],%f46
/*     15 */	fmaddd,s	%f48,%f222,%f220,%f48


/*     15 */	sxar2
/*     15 */	std,s	%f40,[%xg6+32]
/*     15 */	fmuld,s	%f252,%f54,%f250


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f52,%f234,%f232,%f234
/*     15 */	std,s	%f82,[%xg6+96]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f50,%f212,%f218,%f212
/*     15 */	fmaddd,s	%f52,%f210,%f206,%f210


/*     15 */	sxar2
/*     15 */	std,s	%f32,[%xg6+144]
/*     15 */	fmaddd,s	%f50,%f80,%f90,%f80


/*     15 */	sxar2
/*     15 */	fsubd,s	%f34,%f254,%f254
/*     15 */	std,s	%f58,[%xg6+48]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f228,%f240,%f38,%f240
/*     15 */	ldd,s	[%xg8+288],%f64


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f36,%f78,%f42,%f78
/*     15 */	ldd,s	[%xg8+240],%f60


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f52,%f44,%f46,%f44
/*     15 */	fmaddd,s	%f50,%f208,%f56,%f208


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+96],%f68
/*     15 */	fmaddd,s	%f52,%f48,%f246,%f48


/*     15 */	sxar2
/*     15 */	add	%xg8,640,%xg5
/*     15 */	ldd,s	[%xg7+16],%f70


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f50,%f234,%f248,%f234
/*     15 */	fmaddd,s	%f52,%f222,%f220,%f52


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+752],%f32
/*     15 */	fmaddd,s	%f36,%f212,%f244,%f212


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f50,%f210,%f232,%f210
/*     15 */	ldd,s	[%xg8+592],%f98


/*     15 */	sxar2
/*     15 */	fmuld,s	%f204,%f254,%f58
/*     15 */	ldd,s	[%xg8+128],%f72


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f36,%f80,%f218,%f80
/*     15 */	ldd,s	[%xg8+304],%f76


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f202,%f64,%f60,%f62
/*     15 */	ldd,s	[%xg8+256],%f74


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+544],%f92
/*     15 */	fmaddd,s	%f228,%f242,%f238,%f66


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+192],%f86
/*     15 */	fmaddd,s	%f50,%f44,%f68,%f44


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+64],%f100
/*     15 */	fmaddd,s	%f36,%f208,%f70,%f208


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+32],%f90
/*     15 */	fmaddd,s	%f50,%f48,%f46,%f48


/*     15 */	sxar2
/*     15 */	std,s	%f78,[%xg6+160]
/*     15 */	fmaddd,s	%f36,%f234,%f56,%f234


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f50,%f52,%f246,%f50
/*     15 */	std,s	%f212,[%xg6+224]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f250,%f240,%f72,%f240
/*     15 */	fmaddd,s	%f36,%f210,%f248,%f210


/*     15 */	sxar2
/*     15 */	std,s	%f80,[%xg6+272]
/*     15 */	fmaddd,s	%f202,%f76,%f74,%f202


/*     15 */	sxar2
/*     15 */	fmuld,s	%f226,%f254,%f78
/*     15 */	fmuld,s	%f230,%f254,%f84


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f228,%f62,%f86,%f62
/*     15 */	ldd,s	[%xg8+208],%f102


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+496],%f108
/*     15 */	ldd,s	[%xg8+144],%f104


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f228,%f64,%f60,%f88
/*     15 */	fmaddd,s	%f36,%f44,%f90,%f44


/*     15 */	sxar2
/*     15 */	std,s	%f208,[%xg6+176]
/*     15 */	fmaddd,s	%f250,%f66,%f38,%f66


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f36,%f48,%f68,%f48
/*     15 */	std,s	%f234,[%xg6+240]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f250,%f242,%f238,%f242
/*     15 */	fmaddd,s	%f36,%f50,%f46,%f36


/*     15 */	sxar2
/*     15 */	std,s	%f210,[%xg6+288]
/*     15 */	fmaddd,s	%f58,%f98,%f92,%f94


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f224,%f240,%f100,%f240
/*     15 */	ldd,s	[%xg8],%f110


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f228,%f202,%f102,%f202
/*     15 */	ldd,s	[%xg7+48],%f68


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+80],%f114
/*     15 */	fmaddd,s	%f250,%f62,%f104,%f62


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+160],%f112
/*     15 */	fmaddd,s	%f228,%f76,%f74,%f228


/*     15 */	sxar2
/*     15 */	std,s	%f44,[%xg6+192]
/*     15 */	fmuld,s	%f252,%f254,%f106


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f250,%f88,%f86,%f88
/*     15 */	std,s	%f48,[%xg6+256]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f224,%f66,%f72,%f66
/*     15 */	fmaddd,s	%f250,%f64,%f60,%f64


/*     15 */	sxar2
/*     15 */	std,s	%f36,[%xg6+304]
/*     15 */	fmaddd,s	%f224,%f242,%f38,%f242


/*     15 */	sxar2
/*     15 */	fsubd,s	%f34,%f32,%f32
/*     15 */	std,s	%f68,[%xg6+208]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f84,%f94,%f108,%f94
/*     15 */	ldd,s	[%xg8+608],%f120


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f54,%f240,%f110,%f240
/*     15 */	ldd,s	[%xg8+560],%f116


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f250,%f202,%f112,%f202
/*     15 */	fmaddd,s	%f224,%f62,%f114,%f62


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+96],%f124
/*     15 */	fmaddd,s	%f250,%f228,%f102,%f228


/*     15 */	sxar2
/*     15 */	add	%xg8,960,%xg7
/*     15 */	ldd,s	[%xg8+16],%f126


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f224,%f88,%f104,%f88
/*     15 */	fmaddd,s	%f250,%f76,%f74,%f250


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+1072],%f36
/*     15 */	fmaddd,s	%f54,%f66,%f100,%f66


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f224,%f64,%f86,%f64
/*     15 */	ldd,s	[%xg8+912],%f60


/*     15 */	sxar2
/*     15 */	fmuld,s	%f204,%f32,%f40
/*     15 */	ldd,s	[%xg8+448],%f128


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f54,%f242,%f72,%f242
/*     15 */	ldd,s	[%xg8+624],%f132


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f58,%f120,%f116,%f118
/*     15 */	ldd,s	[%xg8+576],%f130


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+864],%f140
/*     15 */	fmaddd,s	%f84,%f98,%f92,%f122


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+512],%f134
/*     15 */	fmaddd,s	%f224,%f202,%f124,%f202


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+384],%f142
/*     15 */	fmaddd,s	%f54,%f62,%f126,%f62


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+32],%f138
/*     15 */	fmaddd,s	%f224,%f228,%f112,%f228


/*     15 */	sxar2
/*     15 */	std,s	%f240,[%xg6+320]
/*     15 */	fmaddd,s	%f54,%f88,%f114,%f88


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f224,%f250,%f102,%f224
/*     15 */	std,s	%f66,[%xg6+384]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f106,%f94,%f128,%f94
/*     15 */	fmaddd,s	%f54,%f64,%f104,%f64


/*     15 */	sxar2
/*     15 */	std,s	%f242,[%xg6+432]
/*     15 */	fmaddd,s	%f58,%f132,%f130,%f58


/*     15 */	sxar2
/*     15 */	fmuld,s	%f226,%f32,%f46
/*     15 */	fmuld,s	%f230,%f32,%f82


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f84,%f118,%f134,%f118
/*     15 */	ldd,s	[%xg8+528],%f144


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+816],%f148
/*     15 */	ldd,s	[%xg8+464],%f146


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f84,%f120,%f116,%f136
/*     15 */	fmaddd,s	%f54,%f202,%f138,%f202


/*     15 */	sxar2
/*     15 */	std,s	%f62,[%xg6+336]
/*     15 */	fmaddd,s	%f106,%f122,%f108,%f122


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f54,%f228,%f124,%f228
/*     15 */	std,s	%f88,[%xg6+400]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f106,%f98,%f92,%f98
/*     15 */	fmaddd,s	%f54,%f224,%f112,%f54


/*     15 */	sxar2
/*     15 */	std,s	%f64,[%xg6+448]
/*     15 */	fmaddd,s	%f40,%f60,%f140,%f56


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f78,%f94,%f142,%f94
/*     15 */	ldd,s	[%xg8+320],%f150


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f84,%f58,%f144,%f58
/*     15 */	ldd,s	[%xg8+48],%f70


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+400],%f154
/*     15 */	fmaddd,s	%f106,%f118,%f146,%f118


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+480],%f152
/*     15 */	fmaddd,s	%f84,%f132,%f130,%f84


/*     15 */	sxar2
/*     15 */	std,s	%f202,[%xg6+352]
/*     15 */	fmuld,s	%f252,%f32,%f42


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f106,%f136,%f134,%f136
/*     15 */	std,s	%f228,[%xg6+416]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f78,%f122,%f128,%f122
/*     15 */	fmaddd,s	%f106,%f120,%f116,%f120


/*     15 */	sxar2
/*     15 */	std,s	%f54,[%xg6+464]
/*     15 */	fmaddd,s	%f78,%f98,%f108,%f98


/*     15 */	sxar2
/*     15 */	fsubd,s	%f34,%f36,%f36
/*     15 */	std,s	%f70,[%xg6+368]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f82,%f56,%f148,%f56
/*     15 */	ldd,s	[%xg8+928],%f66


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f254,%f94,%f150,%f94
/*     15 */	ldd,s	[%xg8+880],%f156


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f106,%f58,%f152,%f58
/*     15 */	fmaddd,s	%f78,%f118,%f154,%f118


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+416],%f158
/*     15 */	fmaddd,s	%f106,%f84,%f144,%f84


/*     15 */	sxar2
/*     15 */	add	%xg8,1280,%xg8
/*     15 */	ldd,s	[%xg8+-944],%f160


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f78,%f136,%f146,%f136
/*     15 */	fmaddd,s	%f106,%f132,%f130,%f106


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+112],%f54
/*     15 */	fmaddd,s	%f254,%f122,%f142,%f122


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f78,%f120,%f134,%f120
/*     15 */	ldd,s	[%xg7+272],%f80


/*     15 */	sxar2
/*     15 */	fmuld,s	%f204,%f36,%f44
/*     15 */	ldd,s	[%xg5+128],%f92


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f254,%f98,%f128,%f98
/*     15 */	ldd,s	[%xg5+304],%f72


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f40,%f66,%f156,%f62
/*     15 */	ldd,s	[%xg5+256],%f70


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+224],%f76
/*     15 */	fmaddd,s	%f82,%f60,%f140,%f68


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg5+192],%f86
/*     15 */	fmaddd,s	%f78,%f58,%f158,%f58


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg5+64],%f96
/*     15 */	fmaddd,s	%f254,%f118,%f160,%f118


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+-928],%f162
/*     15 */	fmaddd,s	%f78,%f84,%f152,%f84


/*     15 */	sxar2
/*     15 */	std,s	%f94,[%xg6+480]
/*     15 */	fmaddd,s	%f254,%f136,%f154,%f136


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f78,%f106,%f144,%f78
/*     15 */	std,s	%f122,[%xg6+544]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f42,%f56,%f92,%f56
/*     15 */	fmaddd,s	%f254,%f120,%f146,%f120


/*     15 */	sxar2
/*     15 */	std,s	%f98,[%xg6+592]
/*     15 */	fmaddd,s	%f40,%f72,%f70,%f40


/*     15 */	sxar2
/*     15 */	fmuld,s	%f226,%f36,%f50
/*     15 */	fmuld,s	%f230,%f36,%f48


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f82,%f62,%f86,%f62
/*     15 */	ldd,s	[%xg5+208],%f88


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+176],%f90
/*     15 */	ldd,s	[%xg5+144],%f94


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f82,%f66,%f156,%f74
/*     15 */	fmaddd,s	%f254,%f58,%f162,%f58


/*     15 */	sxar2
/*     15 */	std,s	%f118,[%xg6+496]
/*     15 */	fmaddd,s	%f42,%f68,%f148,%f68


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f254,%f84,%f158,%f84
/*     15 */	std,s	%f136,[%xg6+560]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f42,%f60,%f140,%f60
/*     15 */	fmaddd,s	%f254,%f78,%f152,%f254


/*     15 */	sxar2
/*     15 */	std,s	%f120,[%xg6+608]
/*     15 */	fmaddd,s	%f44,%f80,%f76,%f78


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f46,%f56,%f96,%f56
/*     15 */	ldd,s	[%xg5],%f176


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f82,%f40,%f88,%f40
/*     15 */	ldd,s	[%xg8+-912],%f98


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg5+80],%f110
/*     15 */	fmaddd,s	%f42,%f62,%f94,%f62


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg5+160],%f106
/*     15 */	fmaddd,s	%f82,%f72,%f70,%f82


/*     15 */	sxar2
/*     15 */	std,s	%f58,[%xg6+512]
/*     15 */	fmuld,s	%f252,%f36,%f52


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f42,%f74,%f86,%f74
/*     15 */	std,s	%f84,[%xg6+576]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f46,%f68,%f92,%f68
/*     15 */	fmaddd,s	%f42,%f66,%f156,%f66


/*     15 */	sxar2
/*     15 */	std,s	%f254,[%xg6+624]
/*     15 */	fmaddd,s	%f46,%f60,%f148,%f60

/*     15 */	add	%o0,4,%o0


/*     15 */	sxar2
/*     15 */	fsubd,s	%f34,%f54,%f54
/*     15 */	std,s	%f98,[%xg6+528]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f48,%f78,%f90,%f78
/*     15 */	add	%xg6,640,%xg6


/*     15 */	sxar2
/*     15 */	sub	%xg0,4,%xg0
/*     15 */	cmp	%xg0,9

/*     15 */	bge,pt	%icc, .L5716
	nop


.L5895:


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+288],%f116
/*    ??? */	ldd,s	[%fp+1535],%f238


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f48,%f80,%f76,%f102
/*     15 */	fmaddd,s	%f52,%f80,%f76,%f80


/*     15 */	sxar2
/*    ??? */	ldd,s	[%fp+1551],%f240
/*     15 */	ldd,s	[%xg7+240],%f112


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f42,%f40,%f106,%f40
/*     15 */	fmaddd,s	%f42,%f82,%f88,%f82


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+272],%f130
/*     15 */	fmaddd,s	%f42,%f72,%f70,%f42


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+224],%f126
/*     15 */	fmaddd,s	%f46,%f62,%f110,%f62


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+288],%f138
/*     15 */	ldd,s	[%xg8+240],%f134


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f46,%f74,%f94,%f74
/*     15 */	fmaddd,s	%f46,%f66,%f86,%f66


/*     15 */	sxar2
/*     15 */	add	%xg8,320,%xg9
/*     15 */	fmuld,s	%f238,%f54,%f98


/*     15 */	sxar2
/*    ??? */	ldd,s	[%fp+1503],%f242
/*     15 */	fmaddd,s	%f32,%f56,%f176,%f56


/*     15 */	sxar2
/*     15 */	fmuld,s	%f240,%f54,%f100
/*     15 */	ldd,s	[%xg7+304],%f122

/*     15 */	sxar1
/*     15 */	fmaddd,s	%f44,%f116,%f112,%f114

/*     15 */	add	%o0,3,%o0


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+256],%f120
/*     15 */	ldd,s	[%xg8+304],%f144


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f48,%f116,%f112,%f118
/*     15 */	fmaddd,s	%f52,%f102,%f90,%f102


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+192],%f148
/*     15 */	ldd,s	[%xg8+256],%f142


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f52,%f116,%f112,%f116
/*     15 */	fmaddd,s	%f46,%f82,%f106,%f82


/*     15 */	sxar2
/*     15 */	fmuld,s	%f242,%f54,%f104
/*     15 */	ldd,s	[%xg8+176],%f152


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+192],%f156
/*     15 */	fmaddd,s	%f50,%f80,%f90,%f80


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+208],%f150
/*     15 */	ldd,s	[%xg8+208],%f158


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f32,%f68,%f96,%f68
/*     15 */	fmaddd,s	%f32,%f60,%f92,%f60


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f98,%f130,%f126,%f128
/*     15 */	sub	%xg0,3,%xg0


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f98,%f138,%f134,%f136
/*     15 */	ldd,s	[%xg7+128],%f154


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f44,%f122,%f120,%f44
/*     15 */	fmaddd,s	%f48,%f122,%f120,%f124


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+144],%f160
/*     15 */	fmaddd,s	%f100,%f130,%f126,%f132


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f100,%f138,%f134,%f140
/*     15 */	ldd,s	[%xg8+128],%f164


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f98,%f144,%f142,%f98
/*     15 */	fmaddd,s	%f100,%f144,%f142,%f146


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+144],%f166
/*    ??? */	ldd,s	[%fp+1519],%f244


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f48,%f114,%f148,%f114
/*     15 */	fmaddd,s	%f52,%f118,%f148,%f118


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+160],%f162
/*     15 */	fmaddd,s	%f52,%f122,%f120,%f122


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f52,%f78,%f154,%f78
/*     15 */	fmaddd,s	%f100,%f128,%f152,%f128


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f100,%f136,%f156,%f136
/*     15 */	ldd,s	[%xg8+160],%f168


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f48,%f44,%f150,%f48
/*     15 */	fmaddd,s	%f52,%f124,%f150,%f124


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg5+96],%f170
/*     15 */	fmaddd,s	%f104,%f132,%f152,%f132


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+64],%f172
/*     15 */	fmaddd,s	%f104,%f130,%f126,%f130


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f100,%f98,%f158,%f100
/*     15 */	fmaddd,s	%f104,%f140,%f156,%f140


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+80],%f174
/*     15 */	fmaddd,s	%f52,%f114,%f160,%f114


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f104,%f138,%f134,%f138
/*     15 */	ldd,s	[%xg7+96],%f178


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f104,%f146,%f158,%f146
/*     15 */	ldd,s	[%xg8+64],%f180


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f104,%f144,%f142,%f144
/*     15 */	fmaddd,s	%f104,%f128,%f164,%f128


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+80],%f182
/*     15 */	fmaddd,s	%f104,%f136,%f166,%f136


/*     15 */	sxar2
/*     15 */	fmuld,s	%f244,%f54,%f108
/*     15 */	fmaddd,s	%f52,%f48,%f162,%f52


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+96],%f184
/*     15 */	ldd,s	[%xg5+16],%f186


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f46,%f40,%f170,%f40
/*     15 */	fmaddd,s	%f50,%f78,%f172,%f78


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f104,%f100,%f168,%f104
/*     15 */	ldd,s	[%xg5+32],%f188


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f50,%f102,%f154,%f102
/*     15 */	ldd,s	[%xg7],%f190


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f50,%f114,%f174,%f114
/*     15 */	fmaddd,s	%f50,%f118,%f160,%f118


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+16],%f192
/*     15 */	fmaddd,s	%f50,%f116,%f148,%f116


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f50,%f124,%f162,%f124
/*     15 */	fmaddd,s	%f46,%f42,%f88,%f46


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+32],%f194
/*     15 */	ldd,s	[%xg8],%f196


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f32,%f62,%f186,%f62
/*     15 */	fmaddd,s	%f50,%f52,%f178,%f52


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f108,%f128,%f180,%f128
/*     15 */	ldd,s	[%xg8+16],%f198


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f108,%f132,%f164,%f132
/*     15 */	fmaddd,s	%f108,%f130,%f152,%f130


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+32],%f200
/*     15 */	fmaddd,s	%f108,%f136,%f182,%f136


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg5+48],%f246
/*     15 */	fmaddd,s	%f108,%f140,%f166,%f140


/*     15 */	sxar2
/*     15 */	mov	%xg9,%xg5
/*     15 */	fmaddd,s	%f108,%f138,%f156,%f138


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+48],%f248
/*     15 */	fmaddd,s	%f108,%f104,%f184,%f104


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f50,%f122,%f150,%f50
/*     15 */	fmaddd,s	%f108,%f146,%f168,%f146


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+48],%f250
/*     15 */	fmaddd,s	%f108,%f144,%f158,%f108


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f32,%f74,%f110,%f74
/*     15 */	fmaddd,s	%f32,%f66,%f94,%f66


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f32,%f40,%f188,%f40
/*     15 */	fmaddd,s	%f32,%f82,%f170,%f82


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f36,%f78,%f190,%f78
/*     15 */	fmaddd,s	%f36,%f102,%f172,%f102


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f36,%f80,%f154,%f80
/*     15 */	fmaddd,s	%f36,%f114,%f192,%f114


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f36,%f118,%f174,%f118
/*     15 */	fmaddd,s	%f36,%f116,%f160,%f116


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f36,%f52,%f194,%f52
/*     15 */	fmaddd,s	%f36,%f124,%f178,%f124


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f54,%f128,%f196,%f128
/*     15 */	fmaddd,s	%f54,%f132,%f180,%f132


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f54,%f130,%f164,%f130
/*     15 */	fmaddd,s	%f54,%f136,%f198,%f136


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f54,%f140,%f182,%f140
/*     15 */	fmaddd,s	%f54,%f138,%f166,%f138


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f54,%f104,%f200,%f104
/*     15 */	fmaddd,s	%f32,%f46,%f106,%f32


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f54,%f146,%f184,%f146
/*     15 */	fmaddd,s	%f36,%f50,%f162,%f36


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f54,%f108,%f168,%f54
/*     15 */	std,s	%f56,[%xg6]


/*     15 */	sxar2
/*     15 */	std,s	%f68,[%xg6+64]
/*     15 */	std,s	%f60,[%xg6+112]


/*     15 */	sxar2
/*     15 */	std,s	%f62,[%xg6+16]
/*     15 */	std,s	%f74,[%xg6+80]


/*     15 */	sxar2
/*     15 */	std,s	%f66,[%xg6+128]
/*     15 */	std,s	%f40,[%xg6+32]


/*     15 */	sxar2
/*     15 */	std,s	%f82,[%xg6+96]
/*     15 */	std,s	%f32,[%xg6+144]


/*     15 */	sxar2
/*     15 */	std,s	%f246,[%xg6+48]
/*     15 */	std,s	%f78,[%xg6+160]


/*     15 */	sxar2
/*     15 */	std,s	%f102,[%xg6+224]
/*     15 */	std,s	%f80,[%xg6+272]


/*     15 */	sxar2
/*     15 */	std,s	%f114,[%xg6+176]
/*     15 */	std,s	%f118,[%xg6+240]


/*     15 */	sxar2
/*     15 */	std,s	%f116,[%xg6+288]
/*     15 */	std,s	%f52,[%xg6+192]


/*     15 */	sxar2
/*     15 */	std,s	%f124,[%xg6+256]
/*     15 */	std,s	%f36,[%xg6+304]


/*     15 */	sxar2
/*     15 */	std,s	%f248,[%xg6+208]
/*     15 */	std,s	%f128,[%xg6+320]


/*     15 */	sxar2
/*     15 */	std,s	%f132,[%xg6+384]
/*     15 */	std,s	%f130,[%xg6+432]


/*     15 */	sxar2
/*     15 */	std,s	%f136,[%xg6+336]
/*     15 */	std,s	%f140,[%xg6+400]


/*     15 */	sxar2
/*     15 */	std,s	%f138,[%xg6+448]
/*     15 */	std,s	%f104,[%xg6+352]


/*     15 */	sxar2
/*     15 */	std,s	%f146,[%xg6+416]
/*     15 */	std,s	%f54,[%xg6+464]


/*     15 */	sxar2
/*     15 */	std,s	%f250,[%xg6+368]
/*     15 */	add	%xg6,480,%xg6

.L5890:


.L5889:


.L5892:


/*     19 */	sxar2
/*     19 */	ldd,s	[%xg5+112],%f164
/*     19 */	ldd,s	[%xg5+272],%f178



/*     21 */	sxar2
/*     21 */	subcc	%xg0,1,%xg0
/*     21 */	ldd,s	[%xg5+224],%f174


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg5+176],%f180
/*     21 */	ldd,s	[%xg5+128],%f182


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg5+64],%f184
/*     21 */	ldd,s	[%xg5],%f186


/*     35 */	sxar2
/* #00001 */	ldd,s	[%fp+1535],%f230
/*     35 */	fsubd,s	%f34,%f164,%f164


/*     38 */	sxar2
/* #00001 */	ldd,s	[%fp+1551],%f228
/* #00001 */	ldd,s	[%fp+1519],%f232


/*     19 */	sxar2
/* #00001 */	ldd,s	[%fp+1503],%f234
/*     19 */	ldd,s	[%xg5+288],%f194


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg5+240],%f190
/*     21 */	ldd,s	[%xg5+192],%f196


/*     38 */	sxar2
/*     38 */	ldd,s	[%xg5+304],%f208
/*     38 */	fmuld,s	%f228,%f164,%f166


/*     21 */	sxar2
/*     21 */	fmuld,s	%f230,%f164,%f168
/*     21 */	ldd,s	[%xg5+256],%f206


/*     38 */	sxar2
/*     38 */	fmuld,s	%f232,%f164,%f170
/*     38 */	fmuld,s	%f234,%f164,%f172


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg5+144],%f198
/*     21 */	ldd,s	[%xg5+80],%f200


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg5+208],%f212
/*     21 */	ldd,s	[%xg5+16],%f202


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg5+160],%f214
/*     21 */	ldd,s	[%xg5+96],%f216


/*     19 */	sxar2
/*     19 */	ldd,s	[%xg5+32],%f218
/*     19 */	ldd,s	[%xg5+48],%f236


/*     21 */	sxar2
/*     21 */	add	%xg5,320,%xg5
/*     21 */	fmaddd,s	%f168,%f178,%f174,%f176


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f168,%f194,%f190,%f192
/*     21 */	fmaddd,s	%f166,%f178,%f174,%f188


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f166,%f194,%f190,%f204
/*     21 */	fmaddd,s	%f168,%f208,%f206,%f168


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f166,%f208,%f206,%f210
/*     21 */	fmaddd,s	%f172,%f178,%f174,%f178


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f172,%f194,%f190,%f194
/*     21 */	fmaddd,s	%f172,%f208,%f206,%f208


/*     21 */	sxar2
/*     21 */	std,s	%f236,[%xg6+48]
/*     21 */	fmaddd,s	%f166,%f176,%f180,%f176


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f166,%f192,%f196,%f192
/*     21 */	fmaddd,s	%f172,%f188,%f180,%f188


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f172,%f204,%f196,%f204
/*     21 */	fmaddd,s	%f166,%f168,%f212,%f166


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f172,%f210,%f212,%f210
/*     21 */	fmaddd,s	%f170,%f178,%f180,%f178


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f170,%f194,%f196,%f194
/*     21 */	fmaddd,s	%f170,%f208,%f212,%f208


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f172,%f176,%f182,%f176
/*     21 */	fmaddd,s	%f172,%f192,%f198,%f192


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f170,%f188,%f182,%f188
/*     21 */	fmaddd,s	%f170,%f204,%f198,%f204


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f172,%f166,%f214,%f172
/*     21 */	fmaddd,s	%f170,%f210,%f214,%f210


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f164,%f178,%f182,%f178
/*     21 */	fmaddd,s	%f164,%f194,%f198,%f194


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f164,%f208,%f214,%f208
/*     21 */	fmaddd,s	%f170,%f176,%f184,%f176


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f170,%f192,%f200,%f192
/*     21 */	fmaddd,s	%f164,%f188,%f184,%f188


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f164,%f204,%f200,%f204
/*     21 */	fmaddd,s	%f170,%f172,%f216,%f170


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f164,%f210,%f216,%f210
/*     21 */	std,s	%f178,[%xg6+112]


/*     21 */	sxar2
/*     21 */	std,s	%f194,[%xg6+128]
/*     21 */	std,s	%f208,[%xg6+144]


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f164,%f176,%f186,%f176
/*     21 */	fmaddd,s	%f164,%f192,%f202,%f192


/*     21 */	sxar2
/*     21 */	std,s	%f188,[%xg6+64]
/*     21 */	fmaddd,s	%f164,%f170,%f218,%f164


/*     21 */	sxar2
/*     21 */	std,s	%f204,[%xg6+80]
/*     21 */	std,s	%f210,[%xg6+96]


/*     21 */	sxar2
/*     21 */	std,s	%f176,[%xg6]
/*     21 */	std,s	%f192,[%xg6+16]


/*     44 */	sxar2
/*     44 */	std,s	%f164,[%xg6+32]
/*     44 */	add	%xg6,160,%xg6

/*     44 */	bne,pt	%icc, .L5892
/*     44 */	add	%o0,1,%o0


.L5888:

/*     44 */
/*     44 */	ba	.L5714
	nop


.L5726:

/*     44 *//*     44 */	call	__mpc_obar
/*     44 */	ldx	[%fp+2199],%o0

/*     44 *//*     44 */	call	__mpc_obar
/*     44 */	ldx	[%fp+2199],%o0


.L5727:

/*     44 */	ret
	restore



.LLFE5:
	.size	_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE._OMP_1,.-_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE._OMP_1
	.type	_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE._OMP_1,#function
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite6-k.cpp _ZN7Gravity19calc_force_in_rangeEiidP5Force $"
	.section	".text"
	.global	_ZN7Gravity19calc_force_in_rangeEiidP5Force
	.align	64
_ZN7Gravity19calc_force_in_rangeEiidP5Force:
.LLFB6:
.L566:

/*     47 */	save	%sp,-2240,%sp
.LLCFI4:
/*     47 */	stw	%i2,[%fp+2195]
/*     47 */	stx	%i0,[%fp+2175]
/*     47 */	stw	%i1,[%fp+2187]
/*     47 */	std	%f6,[%fp+2199]
/*     47 */	stx	%i4,[%fp+2207]

.L567:

/*     52 */	sethi	%h44(.LB0..119.1),%l0

/*     52 */	or	%l0,%m44(.LB0..119.1),%l0

/*     52 */	sllx	%l0,12,%l0

/*     52 */	or	%l0,%l44(.LB0..119.1),%l0


/*     52 */	sxar2
/*     52 */	ldsb	[%l0],%xg0
/*     52 */	cmp	%xg0,%g0

/*     52 */	bne,pt	%icc, .L569
	nop


.L568:


.LLEHB0:
/*     52 */	call	__cxa_guard_acquire
/*     52 */	mov	%l0,%o0
.LLEHE0:


.L5533:

/*     52 */	cmp	%o0,%g0

/*     52 */	be	.L569
	nop


.L570:

/*     53 */	sethi	%h44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%o0

/*     53 */	sethi	%h44(_ZN7Gravity6GForceC1Ev),%o3

/*     53 */	or	%o0,%m44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%o0

/*     53 */	or	%o3,%m44(_ZN7Gravity6GForceC1Ev),%o3

/*     53 */	sllx	%o0,12,%o0

/*     53 */	sllx	%o3,12,%o3

/*     53 */	or	%o0,%l44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%o0

/*     53 */	or	%o3,%l44(_ZN7Gravity6GForceC1Ev),%o3

/*     53 */	sethi	%hi(8192),%o1

/*     53 */	mov	144,%o2


.LLEHB1:
/*     53 */	call	__cxa_vec_ctor
/*     53 */	mov	%g0,%o4
.LLEHE1:


.L5686:

/*     53 */	ba	.L5532
	nop


.L573:

/*     53 */

.L574:


/*     53 */	call	__cxa_guard_abort
/*     53 */	mov	%l0,%o0


.L5531:


.LLEHB2:
/*     53 */	call	_Unwind_Resume
/*     53 */	mov	%i0,%o0


.L5532:


/*     53 */	call	__cxa_guard_release
/*     53 */	mov	%l0,%o0
.LLEHE2:


.L569:

/*     55 *//*     55 */	sethi	%h44(_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_2),%o0
/*     55 */	mov	%fp,%l1
/*     55 */	or	%o0,%m44(_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_2),%o0
/*     55 */	mov	%g0,%l2
/*     55 */	sllx	%o0,12,%o0
/*     55 */	mov	%l1,%o1
/*     55 */	or	%o0,%l44(_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_2),%o0
/*     55 */	call	__mpc_opar
/*     55 */	mov	%l2,%o2

/*    142 */
/*    144 *//*    144 */	sethi	%h44(_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_3),%o0
/*    144 */	mov	%l1,%o1
/*    144 */	or	%o0,%m44(_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_3),%o0
/*    144 */	mov	%l2,%o2
/*    144 */	sllx	%o0,12,%o0
/*    144 */	call	__mpc_opar
/*    144 */	or	%o0,%l44(_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_3),%o0

/*    181 */
/*    181 */	ret
	restore



.L629:


.LLFE6:
	.size	_ZN7Gravity19calc_force_in_rangeEiidP5Force,.-_ZN7Gravity19calc_force_in_rangeEiidP5Force
	.type	_ZN7Gravity19calc_force_in_rangeEiidP5Force,#function
	.section	".gcc_except_table",#alloc
	.align	8
.LLLSDA6:
	.byte	255
	.byte	255
	.byte	1
	.uleb128	.LLLSDACSE6-.LLLSDACSB6
.LLLSDACSB6:
	.uleb128	.LLEHB0-.LLFB6
	.uleb128	.LLEHE0-.LLEHB0
	.uleb128	0x0
	.uleb128	0x0
	.uleb128	.LLEHB1-.LLFB6
	.uleb128	.LLEHE1-.LLEHB1
	.uleb128	.L573-.LLFB6
	.uleb128	0x0
	.uleb128	.LLEHB2-.LLFB6
	.uleb128	.LLEHE2-.LLEHB2
	.uleb128	0x0
	.uleb128	0x0
.LLLSDACSE6:
	.sleb128	0
	.sleb128	0
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite6-k.cpp _ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_2 $"
	.section	".text"
	.align	64
_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_2:
.LLFB7:
.L5728:

/*     55 */	sethi	%hi(7168),%g1
	xor	%g1,-256,%g1
	save	%sp,%g1,%sp
.LLCFI5:
/*     55 */	stx	%i0,[%fp+2175]
/*     55 */	stx	%i1,[%fp+2183]
/*     55 */	stx	%i2,[%fp+2191]
/*     55 */	stx	%i3,[%fp+2199]
/*     55 */	stx	%i0,[%fp+2175]

.L5729:

/*     55 *//*     55 */	sxar1
/*     55 */	ldsw	[%i0+2031],%xg14
/*     55 */
.LLEHB3:
/*     57 */	call	omp_get_thread_num
	nop
/*     57 */	mov	%o0,%l3

.L5730:

/*     58 */
/*     58 */	call	__mpc_pmnm
	nop
/*     58 */	sxar2
/*     58 */	ldx	[%fp+2191],%xg13
/*     58 */	cmp	%xg13,%o0
/*     58 */	bne,pt	%xcc, .L5736
	nop


.L5731:

/*     58 */
/*     59 */	call	omp_get_num_threads
	nop
.LLEHE3:


.L5732:

/*     59 */	ba	.L5735
	nop


.L5733:


.L5734:

/*      0 */	call	_ZSt9terminatev
	nop


.L5735:

/*     59 */	stw	%o0,[%i0+2027]

.L5736:

/*     59 */
/*     61 */	sxar1
/*     61 */	ldx	[%i0+2175],%xg11

/*     62 */	ldsw	[%i0+2187],%l2

/*     61 */	sxar2
/*     61 */	ldsw	[%i0+2195],%xg12
/*     61 */	ldsw	[%xg11],%l1

/*     62 */	sxar1
/*     62 */	cmp	%l2,%xg12
/*     62 */	bge	.L5754
	nop


.L5737:


/*     62 */	sxar2
/*     62 */	fzero,s	%f160
/*    ??? */	sethi	%hi(5168),%xg10


/*     62 */	sxar2
/*    ??? */	xor	%xg10,-49,%xg10
/*    ??? */	std,s	%f160,[%fp+%xg10]

.L5738:

/*     83 */	sethi	%h44(.LR0.cnt.4),%g1

/*     83 */	sethi	%h44(.LR0.cnt.5),%g2

/*     83 */	or	%g1,%m44(.LR0.cnt.4),%g1

/*     83 */	or	%g2,%m44(.LR0.cnt.5),%g2

/*     83 */	sllx	%g1,12,%g1

/*     83 */	sxar1
/*    ??? */	sethi	%hi(5136),%xg7

/*     83 */	or	%g1,%l44(.LR0.cnt.4),%g1

/*     83 */	sllx	%g2,12,%g2

/*     83 */	or	%g2,%l44(.LR0.cnt.5),%g2


/*     83 */	sxar2
/*     83 */	ldd	[%g1],%f124
/*     83 */	ldd	[%g1],%f380



/*     83 */	sxar2
/*    ??? */	xor	%xg7,-17,%xg7
/*     83 */	sethi	%h44(.LR0.cnt.7),%xg0


/*     83 */	sxar2
/*     83 */	ldd	[%g2],%f126
/*     83 */	ldd	[%g2],%f382



/*     83 */	sxar2
/*    ??? */	sethi	%hi(5152),%xg8
/*     83 */	or	%xg0,%m44(.LR0.cnt.7),%xg0


/*     83 */	sxar2
/*    ??? */	xor	%xg8,-33,%xg8
/*     83 */	sllx	%xg0,12,%xg0


/*     83 */	sxar2
/*    ??? */	sethi	%hi(5184),%xg9
/*     83 */	or	%xg0,%l44(.LR0.cnt.7),%xg0


/*     83 */	sxar2
/*    ??? */	std,s	%f124,[%fp+%xg7]
/*    ??? */	xor	%xg9,-65,%xg9

/*     83 */	srl	%l1,31,%l0

/*     83 */	sra	%l3,%g0,%l3

/*     83 */	sxar1
/*    ??? */	std,s	%f126,[%fp+%xg8]

/*     83 */	add	%l0,%l1,%l0


/*     83 */	sxar2
/*     83 */	ldd	[%xg0],%f128
/*     83 */	ldd	[%xg0],%f384

/*     83 */	sethi	%h44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%l7


/*     83 */	sra	%l0,1,%l0

/*     83 */	sllx	%l3,3,%l4

/*     83 */	add	%l0,%l0,%l5

/*     83 */	or	%l7,%m44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%l7

/*     83 */	add	%l4,%l3,%l4

/*     83 */	mov	1,%i2

/*     83 */	sub	%l1,%l5,%l5

/*     83 */	sllx	%l7,12,%l7

/*     83 */	sxar1
/*    ??? */	std,s	%f128,[%fp+%xg9]

/*     83 */	sllx	%l4,13,%l4

/*     83 */	add	%l0,1,%l6

/*     83 */	or	%l7,%l44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%l7

/*     83 */	sra	%i2,%g0,%i1

.L5739:


/*     72 */	sxar2
/*    ??? */	sethi	%hi(5168),%xg5
/*     72 */	srl	%l2,31,%xg0


/*     34 */	sxar2
/*     34 */	ldd	[%i0+2199],%f140
/*    ??? */	xor	%xg5,-49,%xg5


/*     26 */	sxar2
/*     26 */	add	%xg0,%l2,%xg0
/*     26 */	ldd	[%i0+2199],%f396



/*     72 */	sxar2
/*    ??? */	ldd,s	[%fp+%xg5],%f138
/*     72 */	sra	%xg0,1,%xg0


/*     72 */	sxar2
/*     72 */	sra	%xg0,%g0,%xg0
/*     72 */	sllx	%xg0,2,%xg1


/*     34 */	sxar2
/*     34 */	add	%xg1,%xg0,%xg1
/*     34 */	std,s	%f138,[%fp+-2625]


/*     34 */	sxar2
/*     34 */	sllx	%xg1,5,%xg1
/*     34 */	std,s	%f138,[%fp+-2609]


/*     34 */	sxar2
/*     34 */	std,s	%f138,[%fp+-2593]
/*     34 */	std,s	%f138,[%fp+-2577]


/*     34 */	sxar2
/*     34 */	std,s	%f138,[%fp+-2561]
/*     34 */	std,s	%f138,[%fp+-2545]


/*     34 */	sxar2
/*     34 */	std,s	%f138,[%fp+-2529]
/*     34 */	std,s	%f138,[%fp+-2513]


/*     34 */	sxar2
/*     34 */	std,s	%f138,[%fp+-2497]
/*     34 */	std,s	%f138,[%fp+-2481]


/*     34 */	sxar2
/*     34 */	std,s	%f138,[%fp+-2465]
/*     34 */	std,s	%f138,[%fp+-2449]


/*     34 */	sxar2
/*     34 */	std,s	%f138,[%fp+-2433]
/*     34 */	std,s	%f138,[%fp+-2417]


/*     34 */	sxar2
/*     34 */	std,s	%f138,[%fp+-2401]
/*     34 */	std,s	%f138,[%fp+-2385]


/*     34 */	sxar2
/*     34 */	std,s	%f138,[%fp+-2369]
/*     34 */	std,s	%f138,[%fp+-2353]


/*     72 */	sxar2
/*     72 */	ldx	[%i0+2175],%xg6
/*     72 */	ldx	[%xg6+16],%xg2


/*     19 */	sxar2
/*     19 */	add	%xg2,%xg1,%xg2
/*     19 */	ldd,s	[%xg2],%f142


/*     19 */	sxar2
/*     19 */	std,s	%f142,[%fp+-2337]
/*     19 */	ldd,s	[%xg2+16],%f144


/*     19 */	sxar2
/*     19 */	std,s	%f144,[%fp+-2321]
/*     19 */	ldd,s	[%xg2+32],%f146


/*     19 */	sxar2
/*     19 */	std,s	%f146,[%fp+-2305]
/*     19 */	ldd,s	[%xg2+64],%f148


/*     19 */	sxar2
/*     19 */	std,s	%f148,[%fp+-2289]
/*     19 */	ldd,s	[%xg2+80],%f150


/*     19 */	sxar2
/*     19 */	std,s	%f150,[%fp+-2273]
/*     19 */	ldd,s	[%xg2+96],%f152


/*     19 */	sxar2
/*     19 */	std,s	%f152,[%fp+-2257]
/*     19 */	ldd,s	[%xg2+112],%f154


/*     19 */	sxar2
/*     19 */	std,s	%f154,[%fp+-2241]
/*     19 */	ldd,s	[%xg2+128],%f156


/*     19 */	sxar2
/*     19 */	std,s	%f156,[%fp+-2225]
/*     19 */	ldd,s	[%xg2+144],%f158


/*     19 */	sxar2
/*     19 */	std,s	%f140,[%fp+-2193]
/*     19 */	std,s	%f158,[%fp+-2209]

/*     82 */
/*     82 */
/*     83 */	cmp	%l1,%g0
/*     83 */	ble	.L5752
	nop


.L5740:

/*     83 */	cmp	%l5,%g0

/*     83 */	sxar1
/*     83 */	mov	%l0,%xg4

/*     83 */	be	.L5742
	nop


.L5741:

/*     83 */	sxar1
/*     83 */	mov	%l6,%xg4

.L5742:


/*     83 */	sxar2
/*     83 */	ldx	[%fp+2183],%xg5
/*     83 */	ldx	[%fp+2191],%xg6


/*     83 */	sxar2
/*     83 */	sra	%xg4,%g0,%xg3
/*     83 */	sra	%xg5,%g0,%xg5


/*     83 */	sxar2
/*     83 */	sra	%xg6,%g0,%xg6
/*     83 */	sra	%xg5,%g0,%xg7


/*     83 */	sxar2
/*     83 */	sdivx	%xg3,%xg7,%xg3

/*     83 */	sra	%xg3,%g0,%xg3


/*     83 */	sxar2
/*     83 */	mulx	%xg5,%xg3,%xg5
/*     83 */	subcc	%xg4,%xg5,%xg4

/*     83 */	bne,pt	%icc, .L5744
	nop


.L5743:


/*     83 */	sxar2
/*     83 */	add	%xg6,%xg6,%xg6
/*     83 */	add	%xg3,%xg3,%xg8


/*     83 */	sxar2
/*     83 */	mulx	%xg6,%xg3,%xg6
/*     83 */	add	%xg8,%xg6,%xg8

/*     83 */	sxar1
/*     83 */	sub	%xg8,1,%xg8

/*     83 */	ba	.L5747
	nop


.L5744:

/*     83 */	sxar1
/*     83 */	cmp	%xg6,%xg4

/*     83 */	bl	.L5746
	nop


.L5745:


/*     83 */	sxar2
/*     83 */	mulx	%xg6,%xg3,%xg6
/*     83 */	add	%xg3,%xg3,%xg8


/*     83 */	sxar2
/*     83 */	add	%xg6,%xg4,%xg6
/*     83 */	add	%xg6,%xg6,%xg6


/*     83 */	sxar2
/*     83 */	add	%xg8,%xg6,%xg8
/*     83 */	sub	%xg8,1,%xg8

/*     83 */	ba	.L5747
	nop


.L5746:


/*     83 */	sxar2
/*     83 */	add	%xg3,1,%xg3
/*     83 */	add	%xg6,%xg6,%xg6


/*     83 */	sxar2
/*     83 */	mulx	%xg6,%xg3,%xg6
/*     83 */	add	%xg3,%xg3,%xg8


/*     83 */	sxar2
/*     83 */	add	%xg8,%xg6,%xg8
/*     83 */	sub	%xg8,1,%xg8

.L5747:

/*     83 */	sxar1
/*     83 */	cmp	%xg3,%g0

/*     83 */	be	.L5752
	nop


.L5748:


/*     83 */	sxar2
/*     83 */	sub	%xg8,%xg6,%xg8
/*     83 */	ldx	[%i0+2175],%xg4


/*     83 */	sxar2
/*     83 */	ldd,s	[%fp+-2369],%f184
/*     83 */	srl	%xg8,31,%xg9


/*     83 */	sxar2
/*     83 */	ldd,s	[%fp+-2353],%f186
/*     83 */	add	%xg8,%xg9,%xg8


/*     83 */	sxar2
/*     83 */	ldd,s	[%fp+-2401],%f180
/*     83 */	sra	%xg8,1,%xg8


/*     83 */	sxar2
/*     83 */	add	%xg8,1,%xg8
/*     83 */	ldx	[%xg4+16],%xg11


/*     83 */	sxar2
/*     83 */	ldd,s	[%fp+-2385],%f182
/*     83 */	sra	%xg8,%g0,%xg8


/*     83 */	sxar2
/*     83 */	ldd,s	[%fp+-2433],%f176
/*     83 */	sub	%i1,%xg8,%xg8


/*     83 */	sxar2
/*     83 */	ldd,s	[%fp+-2417],%f178
/*     83 */	ldd,s	[%fp+-2465],%f172


/*     83 */	sxar2
/*     83 */	srax	%xg8,32,%xg10
/*     83 */	ldd,s	[%fp+-2449],%f174


/*     83 */	sxar2
/*     83 */	ldd,s	[%fp+-2497],%f168
/*     83 */	and	%xg8,%xg10,%xg8


/*     83 */	sxar2
/*     83 */	add	%xg11,48,%xg12
/*     83 */	ldd,s	[%fp+-2481],%f170


/*     83 */	sxar2
/*     83 */	ldd,s	[%fp+-2529],%f164
/*     83 */	sub	%i2,%xg8,%xg8


/*     83 */	sxar2
/*     83 */	add	%xg11,16,%xg13
/*     83 */	ldd,s	[%fp+-2513],%f166


/*     83 */	sxar2
/*     83 */	ldd,s	[%fp+-2561],%f160
/*     83 */	cmp	%xg8,6


/*     83 */	sxar2
/*     83 */	add	%xg11,32,%xg14
/*     83 */	ldd,s	[%fp+-2545],%f162


/*    195 */	sxar2
/*    195 */	ldd,s	[%fp+-2593],%f156
/*    195 */	add	%xg11,64,%xg15


/*     83 */	sxar2
/*     83 */	add	%xg11,80,%xg16
/*     83 */	ldd,s	[%fp+-2577],%f158


/*    195 */	sxar2
/*    195 */	ldd,s	[%fp+-2625],%f152
/*    195 */	add	%xg11,96,%xg17


/*     83 */	sxar2
/*     83 */	add	%xg11,112,%xg18
/*     83 */	ldd,s	[%fp+-2609],%f154


/*    195 */	sxar2
/*    195 */	ldd,s	[%fp+-2337],%f34
/*    195 */	add	%xg11,128,%xg19


/*     83 */	sxar2
/*     83 */	add	%xg11,144,%xg20
/*     83 */	ldd,s	[%fp+-2321],%f40


/*     83 */	sxar2
/*     83 */	ldd,s	[%fp+-2305],%f46
/*     83 */	ldd,s	[%fp+-2289],%f70


/*     83 */	sxar2
/*     83 */	ldd,s	[%fp+-2273],%f76
/*     83 */	ldd,s	[%fp+-2257],%f86


/*     83 */	sxar2
/*     83 */	ldd,s	[%fp+-2241],%f118
/*     83 */	ldd,s	[%fp+-2225],%f92


/*     83 */	sxar2
/*     83 */	ldd,s	[%fp+-2209],%f140
/*     83 */	ldd,s	[%fp+-2193],%f64

/*     83 */	bl	.L5900
	nop


.L5896:


.L5903:


/*     83 */	sxar2
/*     83 */	srl	%xg6,31,%xg21
/*    ??? */	sethi	%hi(5136),%xg0


/*     83 */	sxar2
/*     83 */	add	%xg21,%xg6,%xg21
/*    ??? */	xor	%xg0,-17,%xg0


/*     83 */	sxar2
/*     83 */	add	%xg6,2,%xg22
/*     83 */	sra	%xg21,1,%xg21


/*     83 */	sxar2
/*    ??? */	ldd,s	[%fp+%xg0],%f108
/*     83 */	sra	%xg21,%g0,%xg21


/*     83 */	sxar2
/*    ??? */	sethi	%hi(5152),%xg1
/*     83 */	srl	%xg22,31,%xg23


/*     83 */	sxar2
/*     83 */	sllx	%xg21,2,%xg24
/*     83 */	add	%xg24,%xg21,%xg24


/*     83 */	sxar2
/*    ??? */	xor	%xg1,-33,%xg1
/*     83 */	add	%xg23,%xg22,%xg23


/*     83 */	sxar2
/*     83 */	sllx	%xg24,5,%xg24
/*    ??? */	ldd,s	[%fp+%xg1],%f136


/*     83 */	sxar2
/*     83 */	sra	%xg23,1,%xg23
/*     83 */	ldd,s	[%xg24+%xg11],%f36


/*     83 */	sxar2
/*     83 */	ldd,s	[%xg24+%xg13],%f42
/*    ??? */	sethi	%hi(5184),%xg2


/*     83 */	sxar2
/*     83 */	sra	%xg23,%g0,%xg23
/*     83 */	ldd,s	[%xg24+%xg14],%f48


/*     83 */	sxar2
/*     83 */	ldd,s	[%xg24+%xg15],%f72
/*    ??? */	sethi	%hi(5168),%xg3


/*     83 */	sxar2
/*     83 */	sllx	%xg23,2,%xg25
/*     83 */	ldd,s	[%xg24+%xg16],%f78


/*     83 */	sxar2
/*    ??? */	xor	%xg2,-65,%xg2
/*     83 */	add	%xg25,%xg23,%xg25


/*     83 */	sxar2
/*     83 */	ldd,s	[%xg24+%xg17],%f88
/*    ??? */	xor	%xg3,-49,%xg3


/*     83 */	sxar2
/*     83 */	sllx	%xg25,5,%xg25
/*     83 */	fmsubd,sc	%f36,%f108,%f34,%f32


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f292,%f108,%f34,%f36
/*     83 */	ldd,s	[%xg24+%xg19],%f94


/*     83 */	sxar2
/*     83 */	add	%xg6,4,%xg6
/*     83 */	ldd,s	[%xg25+%xg11],%f52


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f42,%f108,%f40,%f38
/*     83 */	fmsubd,sc	%f298,%f108,%f40,%f42


/*     83 */	sxar2
/*     83 */	ldd,s	[%xg25+%xg13],%f56
/*     83 */	fmsubd,sc	%f48,%f108,%f46,%f44


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f304,%f108,%f46,%f48
/*     83 */	ldd,s	[%xg25+%xg14],%f60


/*     83 */	sxar2
/*     83 */	ldd,s	[%xg24+%xg18],%f120
/*     83 */	fmsubd,sc	%f72,%f108,%f70,%f68


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f78,%f108,%f76,%f74
/*     83 */	ldd,s	[%xg24+%xg20],%f142


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f88,%f108,%f86,%f84
/*     83 */	fmsubd,sc	%f94,%f108,%f92,%f90


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f52,%f108,%f34,%f50
/*     83 */	fmsubd,sc	%f308,%f108,%f34,%f52


/*     83 */	sxar2
/*    ??? */	ldd,s	[%fp+%xg2],%f220
/*     83 */	fmaddd,s	%f32,%f32,%f64,%f62


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f36,%f36,%f64,%f66
/*    ??? */	ldd,s	[%fp+%xg3],%f218


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f56,%f108,%f40,%f54
/*     83 */	fmsubd,sc	%f312,%f108,%f40,%f56


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f60,%f108,%f46,%f58
/*     83 */	fmsubd,sc	%f316,%f108,%f46,%f60


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f328,%f108,%f70,%f72
/*     83 */	fmsubd,sc	%f334,%f108,%f76,%f78


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f344,%f108,%f86,%f88
/*     83 */	fmsubd,sc	%f350,%f108,%f92,%f94


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f50,%f50,%f64,%f80
/*     83 */	fmaddd,s	%f52,%f52,%f64,%f82


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f38,%f38,%f62,%f62
/*     83 */	fmaddd,s	%f42,%f42,%f66,%f66


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f54,%f54,%f80,%f80
/*     83 */	fmaddd,s	%f56,%f56,%f82,%f82


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f44,%f44,%f62,%f62
/*     83 */	fmaddd,s	%f48,%f48,%f66,%f66


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f58,%f58,%f80,%f80
/*     83 */	fmaddd,s	%f60,%f60,%f82,%f82


/*     83 */	sxar2
/*     83 */	frsqrtad,s	%f62,%f96
/*     83 */	frsqrtad,s	%f66,%f98


/*     83 */	sxar2
/*     83 */	fmuld,s	%f62,%f136,%f62
/*     83 */	fmuld,s	%f66,%f136,%f66


/*     83 */	sxar2
/*     83 */	fmuld,s	%f96,%f96,%f100
/*     83 */	fmuld,s	%f98,%f98,%f102


/*     83 */	sxar2
/*     83 */	fnmsubd,s	%f62,%f100,%f136,%f100
/*     83 */	fnmsubd,s	%f66,%f102,%f136,%f102


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f96,%f100,%f96,%f96
/*     83 */	fmaddd,s	%f98,%f102,%f98,%f98


/*     83 */	sxar2
/*     83 */	fmuld,s	%f96,%f96,%f104
/*     83 */	fmuld,s	%f98,%f98,%f106


/*     83 */	sxar2
/*     83 */	fnmsubd,s	%f62,%f104,%f136,%f104
/*     83 */	fnmsubd,s	%f66,%f106,%f136,%f106


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f96,%f104,%f96,%f96
/*     83 */	fmaddd,s	%f98,%f106,%f98,%f98

.L5749:


/*     83 */	sxar2
/*     83 */	srl	%xg6,31,%xg27
/*     83 */	fmsubd,sc	%f120,%f108,%f118,%f100


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f376,%f108,%f118,%f120
/*     83 */	add	%xg27,%xg6,%xg27


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f142,%f108,%f140,%f110
/*     83 */	frsqrtad,s	%f80,%f112


/*     83 */	sxar2
/*     83 */	sra	%xg27,1,%xg27
/*     83 */	fmsubd,sc	%f398,%f108,%f140,%f142


/*     83 */	sxar2
/*     83 */	frsqrtad,s	%f82,%f114
/*     83 */	sra	%xg27,%g0,%xg27


/*     83 */	sxar2
/*     83 */	fmuld,s	%f74,%f74,%f116
/*     83 */	fmuld,s	%f78,%f78,%f122


/*     83 */	sxar2
/*     83 */	sllx	%xg27,2,%xg28
/*     83 */	fmuld,s	%f96,%f96,%f124


/*     83 */	sxar2
/*     83 */	fmuld,s	%f98,%f98,%f126
/*     83 */	add	%xg28,%xg27,%xg28


/*     83 */	sxar2
/*     83 */	fmuld,s	%f38,%f90,%f128
/*     83 */	fmuld,s	%f42,%f94,%f130


/*     83 */	sxar2
/*     83 */	sllx	%xg28,5,%xg28
/*     83 */	fmuld,s	%f38,%f74,%f132


/*     83 */	sxar2
/*     83 */	fmuld,s	%f42,%f78,%f134
/*     83 */	ldd,s	[%xg28+%xg11],%f148


/*     83 */	sxar2
/*     83 */	fmuld,s	%f80,%f136,%f80
/*     83 */	fmuld,s	%f112,%f112,%f138


/*     83 */	sxar2
/*     83 */	fmuld,s	%f82,%f136,%f82
/*     83 */	fmuld,s	%f114,%f114,%f144


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f68,%f68,%f116,%f116
/*     83 */	fmaddd,s	%f72,%f72,%f122,%f122


/*     83 */	sxar2
/*     83 */	fnmsubd,s	%f62,%f124,%f136,%f62
/*     83 */	fnmsubd,s	%f66,%f126,%f136,%f66


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f32,%f100,%f128,%f128
/*     83 */	fmaddd,s	%f36,%f120,%f130,%f130


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f148,%f108,%f34,%f146
/*     83 */	fmsubd,sc	%f404,%f108,%f34,%f148


/*     83 */	sxar2
/*     83 */	ldd,s	[%xg28+%xg13],%f188
/*     83 */	fmaddd,s	%f32,%f68,%f132,%f132


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f36,%f72,%f134,%f134
/*     83 */	ldd,s	[%xg28+%xg14],%f192


/*     83 */	sxar2
/*     83 */	fnmsubd,s	%f80,%f138,%f136,%f138
/*     83 */	fnmsubd,s	%f82,%f144,%f136,%f144


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f84,%f84,%f116,%f116
/*     83 */	fmaddd,s	%f88,%f88,%f122,%f122


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f96,%f62,%f96,%f96
/*     83 */	fmaddd,s	%f98,%f66,%f98,%f98


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f44,%f110,%f128,%f128
/*     83 */	fmaddd,s	%f48,%f142,%f130,%f130


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f188,%f108,%f40,%f150
/*     83 */	fmsubd,sc	%f444,%f108,%f40,%f188


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f192,%f108,%f46,%f190
/*     83 */	fmsubd,sc	%f448,%f108,%f46,%f192


/*     83 */	sxar2
/*     83 */	ldd,s	[%xg25+%xg15],%f200
/*     83 */	fmaddd,s	%f112,%f138,%f112,%f112


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f114,%f144,%f114,%f114
/*     83 */	fmaddd,s	%f44,%f84,%f132,%f132


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f48,%f88,%f134,%f134
/*     83 */	ldd,s	[%xg25+%xg16],%f212


/*     83 */	sxar2
/*     83 */	fmuld,s	%f96,%f96,%f194
/*     83 */	fmuld,s	%f98,%f98,%f196


/*     83 */	sxar2
/*     83 */	faddd,s	%f116,%f128,%f116
/*     83 */	faddd,s	%f122,%f130,%f122


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f200,%f108,%f70,%f198
/*     83 */	fmsubd,sc	%f456,%f108,%f70,%f200


/*     83 */	sxar2
/*     83 */	ldd,s	[%xg25+%xg17],%f216
/*     83 */	fmaddd,s	%f146,%f146,%f64,%f202


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f148,%f148,%f64,%f204
/*     83 */	fmuld,s	%f112,%f112,%f206


/*     83 */	sxar2
/*     83 */	fmuld,s	%f114,%f114,%f208
/*     83 */	fmsubd,sc	%f212,%f108,%f76,%f210


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f468,%f108,%f76,%f212
/*     83 */	ldd,s	[%xg24+%xg12],%f222


/*     83 */	sxar2
/*     83 */	fmuld,s	%f132,%f194,%f132
/*     83 */	fmuld,s	%f134,%f196,%f134


/*     83 */	sxar2
/*     83 */	fmuld,s	%f194,%f116,%f116
/*     83 */	fmuld,s	%f196,%f122,%f122


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f216,%f108,%f86,%f214
/*     83 */	fmsubd,sc	%f472,%f108,%f86,%f216


/*     83 */	sxar2
/*     83 */	ldd,s	[%xg25+%xg19],%f230
/*     83 */	fmaddd,s	%f150,%f150,%f202,%f202


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f188,%f188,%f204,%f204
/*     83 */	fnmsubd,s	%f80,%f206,%f136,%f206


/*     83 */	sxar2
/*     83 */	fnmsubd,s	%f82,%f208,%f136,%f208
/*     83 */	fmaddd,sc	%f222,%f96,%f218,%f96


/*     83 */	sxar2
/*     83 */	fmaddd,sc	%f478,%f98,%f218,%f222
/*     83 */	fmuld,s	%f220,%f132,%f224


/*     83 */	sxar2
/*     83 */	fmuld,s	%f220,%f134,%f226
/*     83 */	fmaddd,s	%f132,%f132,%f116,%f132


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f134,%f134,%f122,%f134
/*     83 */	fmsubd,sc	%f230,%f108,%f92,%f228


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f486,%f108,%f92,%f230
/*     83 */	fmaddd,s	%f190,%f190,%f202,%f202


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f192,%f192,%f204,%f204
/*     83 */	fmaddd,s	%f112,%f206,%f112,%f112


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f114,%f208,%f114,%f114
/*     83 */	fmuld,s	%f96,%f194,%f96


/*     83 */	sxar2
/*     83 */	fmuld,s	%f222,%f196,%f222
/*     83 */	faddd,s	%f224,%f224,%f232


/*     83 */	sxar2
/*     83 */	faddd,s	%f226,%f226,%f234
/*     83 */	fmuld,s	%f220,%f132,%f132


/*     83 */	sxar2
/*     83 */	fmuld,s	%f220,%f134,%f134
/*     83 */	fmaddd,s	%f224,%f32,%f68,%f68


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f226,%f36,%f72,%f72
/*     83 */	fmaddd,s	%f224,%f38,%f74,%f74


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f226,%f42,%f78,%f78
/*     83 */	fmaddd,s	%f224,%f44,%f84,%f224


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f226,%f48,%f88,%f226
/*     83 */	fmaddd,s	%f222,%f36,%f154,%f154


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f96,%f32,%f152,%f152
/*     83 */	fmaddd,s	%f222,%f42,%f158,%f158


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f96,%f38,%f156,%f156
/*     83 */	fmaddd,s	%f222,%f48,%f162,%f162


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f96,%f44,%f160,%f160
/*     83 */	fmaddd,s	%f234,%f72,%f120,%f120


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f232,%f68,%f100,%f100
/*     83 */	fmaddd,s	%f234,%f78,%f94,%f94


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f232,%f74,%f90,%f90
/*     83 */	fmaddd,s	%f232,%f224,%f110,%f232


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f234,%f226,%f142,%f234
/*     83 */	fmaddd,s	%f134,%f36,%f120,%f120


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f132,%f32,%f100,%f100
/*     83 */	fmaddd,s	%f134,%f42,%f94,%f94


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f132,%f38,%f90,%f90
/*     83 */	fmaddd,s	%f132,%f44,%f232,%f132


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f134,%f48,%f234,%f134
/*     83 */	fmaddd,s	%f222,%f72,%f166,%f72


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f96,%f68,%f164,%f68
/*     83 */	fmaddd,s	%f222,%f78,%f170,%f78


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f96,%f74,%f168,%f74
/*     83 */	fmaddd,s	%f222,%f226,%f174,%f226


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f96,%f224,%f172,%f224
/*     83 */	fmaddd,s	%f222,%f120,%f178,%f120


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f96,%f100,%f176,%f100
/*     83 */	fmaddd,s	%f222,%f94,%f182,%f94


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f96,%f90,%f180,%f90
/*     83 */	fmaddd,s	%f222,%f134,%f186,%f222


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f96,%f132,%f184,%f96
/*     83 */	ldd,s	[%xg25+%xg18],%f238


/*     83 */	sxar2
/*     83 */	ldd,s	[%xg25+%xg20],%f242
/*     83 */	add	%xg6,2,%xg29


/*     83 */	sxar2
/*     83 */	srl	%xg29,31,%xg30
/*     83 */	fmsubd,sc	%f238,%f108,%f118,%f236


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f494,%f108,%f118,%f238
/*     83 */	add	%xg30,%xg29,%xg30


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f242,%f108,%f140,%f240
/*     83 */	frsqrtad,s	%f202,%f184


/*     83 */	sxar2
/*     83 */	sra	%xg30,1,%xg30
/*     83 */	fmsubd,sc	%f498,%f108,%f140,%f242


/*     83 */	sxar2
/*     83 */	frsqrtad,s	%f204,%f244
/*     83 */	sra	%xg30,%g0,%xg30


/*     83 */	sxar2
/*     83 */	fmuld,s	%f210,%f210,%f246
/*     83 */	fmuld,s	%f212,%f212,%f248


/*     83 */	sxar2
/*     83 */	sllx	%xg30,2,%xg24
/*     83 */	fmuld,s	%f112,%f112,%f250


/*     83 */	sxar2
/*     83 */	fmuld,s	%f114,%f114,%f252
/*     83 */	add	%xg24,%xg30,%xg24


/*     83 */	sxar2
/*     83 */	fmuld,s	%f54,%f228,%f254
/*     83 */	fmuld,s	%f56,%f230,%f62


/*     83 */	sxar2
/*     83 */	sllx	%xg24,5,%xg24
/*     83 */	fmuld,s	%f54,%f210,%f84


/*     83 */	sxar2
/*     83 */	fmuld,s	%f56,%f212,%f88
/*     83 */	ldd,s	[%xg24+%xg11],%f36


/*     83 */	sxar2
/*     83 */	fmuld,s	%f202,%f136,%f202
/*     83 */	fmuld,s	%f184,%f184,%f98


/*     83 */	sxar2
/*     83 */	fmuld,s	%f204,%f136,%f204
/*     83 */	fmuld,s	%f244,%f244,%f102


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f198,%f198,%f246,%f246
/*     83 */	fmaddd,s	%f200,%f200,%f248,%f248


/*     83 */	sxar2
/*     83 */	fnmsubd,s	%f80,%f250,%f136,%f80
/*     83 */	fnmsubd,s	%f82,%f252,%f136,%f82


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f50,%f236,%f254,%f254
/*     83 */	fmaddd,s	%f52,%f238,%f62,%f62


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f36,%f108,%f34,%f32
/*     83 */	fmsubd,sc	%f292,%f108,%f34,%f36


/*     83 */	sxar2
/*     83 */	ldd,s	[%xg24+%xg13],%f42
/*     83 */	fmaddd,s	%f50,%f198,%f84,%f84


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f52,%f200,%f88,%f88
/*     83 */	ldd,s	[%xg24+%xg14],%f48


/*     83 */	sxar2
/*     83 */	fnmsubd,s	%f202,%f98,%f136,%f98
/*     83 */	fnmsubd,s	%f204,%f102,%f136,%f102


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f214,%f214,%f246,%f246
/*     83 */	fmaddd,s	%f216,%f216,%f248,%f248


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f112,%f80,%f112,%f112
/*     83 */	fmaddd,s	%f114,%f82,%f114,%f114


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f58,%f240,%f254,%f254
/*     83 */	fmaddd,s	%f60,%f242,%f62,%f62


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f42,%f108,%f40,%f38
/*     83 */	fmsubd,sc	%f298,%f108,%f40,%f42


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f48,%f108,%f46,%f44
/*     83 */	fmsubd,sc	%f304,%f108,%f46,%f48


/*     83 */	sxar2
/*     83 */	ldd,s	[%xg28+%xg15],%f166
/*     83 */	fmaddd,s	%f184,%f98,%f184,%f184


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f244,%f102,%f244,%f244
/*     83 */	fmaddd,s	%f58,%f214,%f84,%f84


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f60,%f216,%f88,%f88
/*     83 */	ldd,s	[%xg28+%xg16],%f170


/*     83 */	sxar2
/*     83 */	fmuld,s	%f112,%f112,%f104
/*     83 */	fmuld,s	%f114,%f114,%f106


/*     83 */	sxar2
/*     83 */	faddd,s	%f246,%f254,%f246
/*     83 */	faddd,s	%f248,%f62,%f248


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f166,%f108,%f70,%f164
/*     83 */	fmsubd,sc	%f422,%f108,%f70,%f166


/*     83 */	sxar2
/*     83 */	ldd,s	[%xg28+%xg17],%f124
/*     83 */	fmaddd,s	%f32,%f32,%f64,%f62


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f36,%f36,%f64,%f66
/*     83 */	fmuld,s	%f184,%f184,%f110


/*     83 */	sxar2
/*     83 */	fmuld,s	%f244,%f244,%f116
/*     83 */	fmsubd,sc	%f170,%f108,%f76,%f168


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f426,%f108,%f76,%f170
/*     83 */	ldd,s	[%xg25+%xg12],%f126


/*     83 */	sxar2
/*     83 */	fmuld,s	%f84,%f104,%f84
/*     83 */	fmuld,s	%f88,%f106,%f88


/*     83 */	sxar2
/*     83 */	fmuld,s	%f104,%f246,%f246
/*     83 */	fmuld,s	%f106,%f248,%f248


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f124,%f108,%f86,%f122
/*     83 */	fmsubd,sc	%f380,%f108,%f86,%f124


/*     83 */	sxar2
/*     83 */	ldd,s	[%xg28+%xg19],%f182
/*     83 */	fmaddd,s	%f38,%f38,%f62,%f62


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f42,%f42,%f66,%f66
/*     83 */	fnmsubd,s	%f202,%f110,%f136,%f110


/*     83 */	sxar2
/*     83 */	fnmsubd,s	%f204,%f116,%f136,%f116
/*     83 */	fmaddd,sc	%f126,%f112,%f218,%f112


/*     83 */	sxar2
/*     83 */	fmaddd,sc	%f382,%f114,%f218,%f126
/*     83 */	fmuld,s	%f220,%f84,%f128


/*     83 */	sxar2
/*     83 */	fmuld,s	%f220,%f88,%f130
/*     83 */	fmaddd,s	%f84,%f84,%f246,%f84


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f88,%f88,%f248,%f88
/*     83 */	fmsubd,sc	%f182,%f108,%f92,%f180


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f438,%f108,%f92,%f182
/*     83 */	fmaddd,s	%f44,%f44,%f62,%f62


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f48,%f48,%f66,%f66
/*     83 */	fmaddd,s	%f184,%f110,%f184,%f184


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f244,%f116,%f244,%f244
/*     83 */	fmuld,s	%f112,%f104,%f112


/*     83 */	sxar2
/*     83 */	fmuld,s	%f126,%f106,%f126
/*     83 */	faddd,s	%f128,%f128,%f132


/*     83 */	sxar2
/*     83 */	faddd,s	%f130,%f130,%f134
/*     83 */	fmuld,s	%f220,%f84,%f84


/*     83 */	sxar2
/*     83 */	fmuld,s	%f220,%f88,%f88
/*     83 */	fmaddd,s	%f128,%f50,%f198,%f198


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f130,%f52,%f200,%f200
/*     83 */	fmaddd,s	%f128,%f54,%f210,%f210


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f130,%f56,%f212,%f212
/*     83 */	fmaddd,s	%f128,%f58,%f214,%f128


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f130,%f60,%f216,%f130
/*     83 */	fmaddd,s	%f126,%f52,%f154,%f154


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f112,%f50,%f152,%f152
/*     83 */	fmaddd,s	%f126,%f56,%f158,%f158


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f112,%f54,%f156,%f156
/*     83 */	fmaddd,s	%f126,%f60,%f162,%f162


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f112,%f58,%f160,%f160
/*     83 */	fmaddd,s	%f134,%f200,%f238,%f238


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f132,%f198,%f236,%f236
/*     83 */	fmaddd,s	%f134,%f212,%f230,%f230


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f132,%f210,%f228,%f228
/*     83 */	fmaddd,s	%f132,%f128,%f240,%f132


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f134,%f130,%f242,%f134
/*     83 */	fmaddd,s	%f88,%f52,%f238,%f238


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f84,%f50,%f236,%f236
/*     83 */	fmaddd,s	%f88,%f56,%f230,%f230


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f84,%f54,%f228,%f228
/*     83 */	fmaddd,s	%f84,%f58,%f132,%f84


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f88,%f60,%f134,%f88
/*     83 */	fmaddd,s	%f126,%f200,%f72,%f200


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f112,%f198,%f68,%f198
/*     83 */	fmaddd,s	%f126,%f212,%f78,%f212


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f112,%f210,%f74,%f210
/*     83 */	fmaddd,s	%f126,%f130,%f226,%f130


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f112,%f128,%f224,%f128
/*     83 */	fmaddd,s	%f126,%f238,%f120,%f238


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f112,%f236,%f100,%f236
/*     83 */	fmaddd,s	%f126,%f230,%f94,%f230


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f112,%f228,%f90,%f228
/*     83 */	fmaddd,s	%f126,%f88,%f222,%f126


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f112,%f84,%f96,%f112
/*     83 */	ldd,s	[%xg28+%xg18],%f178


/*     83 */	sxar2
/*     83 */	ldd,s	[%xg28+%xg20],%f142
/*     83 */	add	%xg6,4,%xg31


/*     83 */	sxar2
/*     83 */	srl	%xg31,31,%g1
/*     83 */	fmsubd,sc	%f178,%f108,%f118,%f176


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f434,%f108,%f118,%f178
/*     83 */	add	%g1,%xg31,%g1


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f142,%f108,%f140,%f138
/*     83 */	frsqrtad,s	%f62,%f96

/*     83 */	sra	%g1,1,%g1


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f398,%f108,%f140,%f142
/*     83 */	frsqrtad,s	%f66,%f98

/*     83 */	sra	%g1,%g0,%g1


/*     83 */	sxar2
/*     83 */	fmuld,s	%f168,%f168,%f144
/*     83 */	fmuld,s	%f170,%f170,%f194


/*     83 */	sxar2
/*     83 */	sllx	%g1,2,%xg25
/*     83 */	fmuld,s	%f184,%f184,%f196


/*     83 */	sxar2
/*     83 */	fmuld,s	%f244,%f244,%f206
/*     83 */	add	%xg25,%g1,%xg25


/*     83 */	sxar2
/*     83 */	fmuld,s	%f150,%f180,%f208
/*     83 */	fmuld,s	%f188,%f182,%f214


/*     83 */	sxar2
/*     83 */	sllx	%xg25,5,%xg25
/*     83 */	fmuld,s	%f150,%f168,%f216


/*     83 */	sxar2
/*     83 */	fmuld,s	%f188,%f170,%f222
/*     83 */	ldd,s	[%xg25+%xg11],%f52


/*     83 */	sxar2
/*     83 */	fmuld,s	%f62,%f136,%f62
/*     83 */	fmuld,s	%f96,%f96,%f224


/*     83 */	sxar2
/*     83 */	fmuld,s	%f66,%f136,%f66
/*     83 */	fmuld,s	%f98,%f98,%f226


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f164,%f164,%f144,%f144
/*     83 */	fmaddd,s	%f166,%f166,%f194,%f194


/*     83 */	sxar2
/*     83 */	fnmsubd,s	%f202,%f196,%f136,%f202
/*     83 */	fnmsubd,s	%f204,%f206,%f136,%f204


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f146,%f176,%f208,%f208
/*     83 */	fmaddd,s	%f148,%f178,%f214,%f214


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f52,%f108,%f34,%f50
/*     83 */	fmsubd,sc	%f308,%f108,%f34,%f52


/*     83 */	sxar2
/*     83 */	ldd,s	[%xg25+%xg13],%f56
/*     83 */	fmaddd,s	%f146,%f164,%f216,%f216


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f148,%f166,%f222,%f222
/*     83 */	ldd,s	[%xg25+%xg14],%f60


/*     83 */	sxar2
/*     83 */	fnmsubd,s	%f62,%f224,%f136,%f224
/*     83 */	fnmsubd,s	%f66,%f226,%f136,%f226


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f122,%f122,%f144,%f144
/*     83 */	fmaddd,s	%f124,%f124,%f194,%f194


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f184,%f202,%f184,%f184
/*     83 */	fmaddd,s	%f244,%f204,%f244,%f244


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f190,%f138,%f208,%f208
/*     83 */	fmaddd,s	%f192,%f142,%f214,%f214


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f56,%f108,%f40,%f54
/*     83 */	fmsubd,sc	%f312,%f108,%f40,%f56


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f60,%f108,%f46,%f58
/*     83 */	fmsubd,sc	%f316,%f108,%f46,%f60


/*     83 */	sxar2
/*     83 */	ldd,s	[%xg24+%xg15],%f72
/*     83 */	fmaddd,s	%f96,%f224,%f96,%f96


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f98,%f226,%f98,%f98
/*     83 */	fmaddd,s	%f190,%f122,%f216,%f216


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f192,%f124,%f222,%f222
/*     83 */	ldd,s	[%xg24+%xg16],%f78


/*     83 */	sxar2
/*     83 */	fmuld,s	%f184,%f184,%f232
/*     83 */	fmuld,s	%f244,%f244,%f234


/*     83 */	sxar2
/*     83 */	faddd,s	%f144,%f208,%f144
/*     83 */	faddd,s	%f194,%f214,%f194


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f72,%f108,%f70,%f68
/*     83 */	fmsubd,sc	%f328,%f108,%f70,%f72


/*     83 */	sxar2
/*     83 */	ldd,s	[%xg24+%xg17],%f88
/*     83 */	fmaddd,s	%f50,%f50,%f64,%f80


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f52,%f52,%f64,%f82
/*     83 */	fmuld,s	%f96,%f96,%f240


/*     83 */	sxar2
/*     83 */	fmuld,s	%f98,%f98,%f242
/*     83 */	fmsubd,sc	%f78,%f108,%f76,%f74


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f334,%f108,%f76,%f78
/*     83 */	ldd,s	[%xg28+%xg12],%f186


/*     83 */	sxar2
/*     83 */	fmuld,s	%f216,%f232,%f216
/*     83 */	fmuld,s	%f222,%f234,%f222


/*     83 */	sxar2
/*     83 */	fmuld,s	%f232,%f144,%f144
/*     83 */	fmuld,s	%f234,%f194,%f194


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f88,%f108,%f86,%f84
/*     83 */	fmsubd,sc	%f344,%f108,%f86,%f88


/*     83 */	sxar2
/*     83 */	ldd,s	[%xg24+%xg19],%f94
/*     83 */	fmaddd,s	%f54,%f54,%f80,%f80


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f56,%f56,%f82,%f82
/*     83 */	fnmsubd,s	%f62,%f240,%f136,%f240


/*     83 */	sxar2
/*     83 */	fnmsubd,s	%f66,%f242,%f136,%f242
/*     83 */	fmaddd,sc	%f186,%f184,%f218,%f184


/*     83 */	sxar2
/*     83 */	fmaddd,sc	%f442,%f244,%f218,%f186
/*     83 */	fmuld,s	%f220,%f216,%f172


/*     83 */	sxar2
/*     83 */	fmuld,s	%f220,%f222,%f174
/*     83 */	fmaddd,s	%f216,%f216,%f144,%f216


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f222,%f222,%f194,%f222
/*     83 */	fmsubd,sc	%f94,%f108,%f92,%f90


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f350,%f108,%f92,%f94
/*     83 */	fmaddd,s	%f58,%f58,%f80,%f80


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f60,%f60,%f82,%f82
/*     83 */	fmaddd,s	%f96,%f240,%f96,%f96


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f98,%f242,%f98,%f98
/*     83 */	fmuld,s	%f184,%f232,%f184


/*     83 */	sxar2
/*     83 */	fmuld,s	%f186,%f234,%f186
/*     83 */	faddd,s	%f172,%f172,%f244


/*     83 */	sxar2
/*     83 */	faddd,s	%f174,%f174,%f246
/*     83 */	fmuld,s	%f220,%f216,%f216


/*     83 */	sxar2
/*     83 */	fmuld,s	%f220,%f222,%f222
/*     83 */	fmaddd,s	%f172,%f146,%f164,%f164


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f174,%f148,%f166,%f166
/*     83 */	fmaddd,s	%f172,%f150,%f168,%f168


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f174,%f188,%f170,%f170
/*     83 */	fmaddd,s	%f172,%f190,%f122,%f172


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f174,%f192,%f124,%f174
/*     83 */	fmaddd,s	%f186,%f148,%f154,%f154


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f184,%f146,%f152,%f152
/*     83 */	fmaddd,s	%f186,%f188,%f158,%f158


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f184,%f150,%f156,%f156
/*     83 */	fmaddd,s	%f186,%f192,%f162,%f162


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f184,%f190,%f160,%f160
/*     83 */	fmaddd,s	%f246,%f166,%f178,%f178


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f244,%f164,%f176,%f176
/*     83 */	fmaddd,s	%f246,%f170,%f182,%f182


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f244,%f168,%f180,%f180
/*     83 */	fmaddd,s	%f244,%f172,%f138,%f244


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f246,%f174,%f142,%f246
/*     83 */	fmaddd,s	%f222,%f148,%f178,%f178


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f216,%f146,%f176,%f176
/*     83 */	fmaddd,s	%f222,%f188,%f182,%f182


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f216,%f150,%f180,%f180
/*     83 */	fmaddd,s	%f216,%f190,%f244,%f216


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f222,%f192,%f246,%f222
/*     83 */	fmaddd,s	%f186,%f166,%f200,%f166


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f184,%f164,%f198,%f164
/*     83 */	fmaddd,s	%f186,%f170,%f212,%f170


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f184,%f168,%f210,%f168
/*     83 */	fmaddd,s	%f186,%f174,%f130,%f174


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f184,%f172,%f128,%f172
/*     83 */	fmaddd,s	%f186,%f178,%f238,%f178


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f184,%f176,%f236,%f176
/*     83 */	fmaddd,s	%f186,%f182,%f230,%f182


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f184,%f180,%f228,%f180
/*     83 */	fmaddd,s	%f186,%f222,%f126,%f186


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f184,%f216,%f112,%f184
/*     83 */	ldd,s	[%xg24+%xg18],%f120


/*     83 */	sxar2
/*     83 */	ldd,s	[%xg24+%xg20],%f142
/*     83 */	add	%xg6,6,%xg6


/*     83 */	sxar2
/*     83 */	sub	%xg8,3,%xg8
/*     83 */	cmp	%xg8,7

/*     83 */	bge,pt	%icc, .L5749
	nop


.L5904:


/*     83 */	sxar2
/*     83 */	frsqrtad,s	%f80,%f108
/*     83 */	frsqrtad,s	%f82,%f110

/*    ??? */	sethi	%hi(5152),%o3

/*    ??? */	sethi	%hi(5136),%o4

/*     83 */	sxar1
/*     83 */	ldd,s	[%xg25+%xg16],%f146

/*    ??? */	xor	%o3,-33,%o3

/*     83 */	sxar1
/*     83 */	fmuld,s	%f96,%f96,%f112

/*    ??? */	xor	%o4,-17,%o4


/*     83 */	sxar2
/*    ??? */	ldd,s	[%fp+%o3],%f224
/*     83 */	ldd,s	[%xg25+%xg19],%f150


/*     83 */	sxar2
/*     83 */	fmuld,s	%f98,%f98,%f114
/*     83 */	fmuld,s	%f38,%f74,%f134


/*     83 */	sxar2
/*     83 */	ldd,s	[%xg25+%xg15],%f194
/*    ??? */	ldd,s	[%fp+%o4],%f226


/*     83 */	sxar2
/*     83 */	fmuld,s	%f74,%f74,%f122
/*     83 */	fmuld,s	%f42,%f78,%f136


/*     83 */	sxar2
/*     83 */	ldd,s	[%xg25+%xg18],%f198
/*     83 */	fmuld,s	%f78,%f78,%f124


/*     83 */	sxar2
/*     83 */	ldd,s	[%xg25+%xg17],%f218
/*     83 */	fmuld,s	%f38,%f90,%f126


/*     83 */	sxar2
/*     83 */	fmuld,s	%f42,%f94,%f128
/*     83 */	ldd,s	[%xg25+%xg20],%f222

/*     83 */	sxar1
/*     83 */	ldd,s	[%xg24+%xg12],%f230

/*    ??? */	sethi	%hi(5184),%o5


/*     83 */	sxar2
/*     83 */	fmuld,s	%f80,%f224,%f80
/*     83 */	fmuld,s	%f82,%f224,%f82

/*     83 */	sxar1
/*     83 */	ldd,s	[%xg25+%xg12],%f228

/*    ??? */	xor	%o5,-65,%o5


/*     83 */	sxar2
/*     83 */	sub	%xg8,2,%xg8
/*     83 */	fmuld,s	%f108,%f108,%f130

/*     83 */	sxar1
/*     83 */	fmuld,s	%f110,%f110,%f132

/*    ??? */	sethi	%hi(5168),%o7


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f146,%f226,%f76,%f144
/*     83 */	fmsubd,sc	%f150,%f226,%f92,%f148

/*    ??? */	xor	%o7,-49,%o7


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f402,%f226,%f76,%f146
/*     83 */	fmsubd,sc	%f406,%f226,%f92,%f150


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f194,%f226,%f70,%f192
/*     83 */	fmsubd,sc	%f198,%f226,%f118,%f196


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f120,%f226,%f118,%f116
/*     83 */	fnmsubd,s	%f62,%f112,%f224,%f62


/*     83 */	sxar2
/*     83 */	fnmsubd,s	%f66,%f114,%f224,%f66
/*     83 */	fmsubd,sc	%f450,%f226,%f70,%f194


/*     83 */	sxar2
/*     83 */	fnmsubd,s	%f80,%f130,%f224,%f130
/*     83 */	fnmsubd,s	%f82,%f132,%f224,%f132


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f454,%f226,%f118,%f198
/*     83 */	fmsubd,sc	%f218,%f226,%f86,%f216


/*     83 */	sxar2
/*     83 */	fmuld,s	%f146,%f146,%f206
/*     83 */	fmuld,s	%f56,%f150,%f210


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f376,%f226,%f118,%f120
/*     83 */	fmsubd,sc	%f222,%f226,%f140,%f220


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f142,%f226,%f140,%f138
/*     83 */	fmaddd,s	%f32,%f68,%f134,%f134


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f36,%f72,%f136,%f136
/*     83 */	fmaddd,s	%f96,%f62,%f96,%f96


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f108,%f130,%f108,%f108
/*     83 */	fmaddd,s	%f110,%f132,%f110,%f110


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f98,%f66,%f98,%f98
/*     83 */	fmuld,s	%f144,%f144,%f204


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f474,%f226,%f86,%f218
/*     83 */	fmsubd,sc	%f478,%f226,%f140,%f222


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f194,%f194,%f206,%f206
/*     83 */	fmaddd,s	%f52,%f198,%f210,%f210


/*     83 */	sxar2
/*     83 */	fmsubd,sc	%f398,%f226,%f140,%f142
/*     83 */	fmaddd,s	%f68,%f68,%f122,%f122


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f72,%f72,%f124,%f124
/*     83 */	fmaddd,s	%f32,%f116,%f126,%f126


/*     83 */	sxar2
/*     83 */	fmuld,s	%f108,%f108,%f200
/*     83 */	fmuld,s	%f110,%f110,%f202


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f36,%f120,%f128,%f128
/*     83 */	fmaddd,s	%f44,%f84,%f134,%f134


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f48,%f88,%f136,%f136
/*     83 */	fmuld,s	%f96,%f96,%f188


/*     83 */	sxar2
/*     83 */	fmuld,s	%f98,%f98,%f190
/*     83 */	fmaddd,s	%f192,%f192,%f204,%f204


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f218,%f218,%f206,%f206
/*     83 */	fmaddd,s	%f60,%f222,%f210,%f210


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f84,%f84,%f122,%f122
/*     83 */	fmaddd,s	%f88,%f88,%f124,%f124


/*     83 */	sxar2
/*     83 */	fnmsubd,s	%f80,%f200,%f224,%f200
/*     83 */	fnmsubd,s	%f82,%f202,%f224,%f202


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f44,%f138,%f126,%f126
/*     83 */	fmaddd,s	%f48,%f142,%f128,%f128


/*     83 */	sxar2
/*     83 */	fmuld,s	%f54,%f148,%f208
/*     83 */	fmuld,s	%f134,%f188,%f134


/*     83 */	sxar2
/*     83 */	fmuld,s	%f54,%f144,%f212
/*     83 */	fmuld,s	%f136,%f190,%f136


/*     83 */	sxar2
/*    ??? */	ldd,s	[%fp+%o5],%f244
/*     83 */	fmuld,s	%f56,%f146,%f214


/*     83 */	sxar2
/*     83 */	faddd,s	%f206,%f210,%f206
/*     83 */	fmaddd,s	%f216,%f216,%f204,%f204


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f108,%f200,%f108,%f108
/*     83 */	fmaddd,s	%f110,%f202,%f110,%f110


/*     83 */	sxar2
/*     83 */	faddd,s	%f122,%f126,%f122
/*     83 */	faddd,s	%f124,%f128,%f124


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f50,%f196,%f208,%f208
/*     83 */	fmuld,s	%f244,%f134,%f232


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f50,%f192,%f212,%f212
/*     83 */	fmuld,s	%f244,%f136,%f234


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f52,%f194,%f214,%f214
/*    ??? */	ldd,s	[%fp+%o7],%f254


/*     83 */	sxar2
/*     83 */	fmuld,s	%f108,%f108,%f236
/*     83 */	fmuld,s	%f110,%f110,%f238


/*     83 */	sxar2
/*     83 */	fmuld,s	%f188,%f122,%f122
/*     83 */	fmuld,s	%f190,%f124,%f124


/*     83 */	sxar2
/*     83 */	faddd,s	%f232,%f232,%f240
/*     83 */	fmaddd,s	%f232,%f32,%f68,%f68


/*     83 */	sxar2
/*     83 */	faddd,s	%f234,%f234,%f242
/*     83 */	fmaddd,s	%f234,%f36,%f72,%f72


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f232,%f38,%f74,%f74
/*     83 */	fmaddd,s	%f234,%f42,%f78,%f78


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f232,%f44,%f84,%f232
/*     83 */	fmaddd,sc	%f230,%f96,%f254,%f96


/*     83 */	sxar2
/*     83 */	fnmsubd,s	%f80,%f236,%f224,%f80
/*     83 */	fnmsubd,s	%f82,%f238,%f224,%f82


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f134,%f134,%f122,%f134
/*     83 */	fmaddd,s	%f136,%f136,%f124,%f136


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f234,%f48,%f88,%f234
/*     83 */	fmaddd,sc	%f486,%f98,%f254,%f230


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f242,%f72,%f120,%f120
/*     83 */	fmaddd,s	%f240,%f68,%f116,%f116


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f242,%f78,%f94,%f94
/*     83 */	fmaddd,s	%f240,%f74,%f90,%f90


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f58,%f220,%f208,%f208
/*     83 */	fmaddd,s	%f58,%f216,%f212,%f212


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f108,%f80,%f108,%f108
/*     83 */	fmaddd,s	%f110,%f82,%f110,%f110


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f60,%f218,%f214,%f214
/*     83 */	fmuld,s	%f244,%f134,%f134


/*     83 */	sxar2
/*     83 */	fmuld,s	%f244,%f136,%f136
/*     83 */	fmaddd,s	%f240,%f232,%f138,%f240


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f242,%f234,%f142,%f242
/*     83 */	fmuld,s	%f96,%f188,%f96


/*     83 */	sxar2
/*     83 */	fmuld,s	%f230,%f190,%f230
/*     83 */	faddd,s	%f204,%f208,%f204


/*     83 */	sxar2
/*     83 */	fmuld,s	%f108,%f108,%f246
/*     83 */	fmuld,s	%f110,%f110,%f248


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f134,%f32,%f116,%f116
/*     83 */	fmaddd,s	%f134,%f38,%f90,%f90


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f136,%f36,%f120,%f120
/*     83 */	fmaddd,s	%f136,%f42,%f94,%f94


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f134,%f44,%f240,%f134
/*     83 */	fmaddd,s	%f96,%f232,%f172,%f232


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f136,%f48,%f242,%f136
/*     83 */	fmaddd,s	%f230,%f234,%f174,%f234


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f230,%f36,%f154,%f154
/*     83 */	fmaddd,s	%f96,%f32,%f152,%f152


/*     83 */	sxar2
/*     83 */	fmuld,s	%f212,%f246,%f212
/*     83 */	fmuld,s	%f214,%f248,%f214


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f230,%f42,%f158,%f158
/*     83 */	fmaddd,s	%f96,%f38,%f156,%f156


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f230,%f48,%f162,%f162
/*     83 */	fmaddd,s	%f96,%f44,%f160,%f160


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f230,%f72,%f166,%f72
/*     83 */	fmaddd,s	%f96,%f68,%f164,%f68


/*     83 */	sxar2
/*     83 */	fmuld,s	%f246,%f204,%f204
/*     83 */	fmuld,s	%f248,%f206,%f206


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f230,%f78,%f170,%f78
/*     83 */	fmaddd,s	%f96,%f74,%f168,%f74


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f230,%f120,%f178,%f120
/*     83 */	fmaddd,s	%f96,%f116,%f176,%f116


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f230,%f94,%f182,%f94
/*     83 */	fmaddd,s	%f96,%f90,%f180,%f90


/*     83 */	sxar2
/*     83 */	fmuld,s	%f244,%f212,%f172
/*     83 */	fmuld,s	%f244,%f214,%f174


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f230,%f136,%f186,%f230
/*     83 */	fmaddd,s	%f96,%f134,%f184,%f96


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f212,%f212,%f204,%f212
/*     83 */	fmaddd,s	%f214,%f214,%f206,%f214


/*     83 */	sxar2
/*     83 */	fmaddd,sc	%f228,%f108,%f254,%f184
/*     83 */	fmaddd,sc	%f484,%f110,%f254,%f186


/*     83 */	sxar2
/*     83 */	faddd,s	%f172,%f172,%f250
/*     83 */	faddd,s	%f174,%f174,%f252


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f172,%f50,%f192,%f164
/*     83 */	fmaddd,s	%f174,%f52,%f194,%f166


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f172,%f54,%f144,%f168
/*     83 */	fmaddd,s	%f174,%f56,%f146,%f170


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f172,%f58,%f216,%f172
/*     83 */	fmaddd,s	%f174,%f60,%f218,%f174


/*     83 */	sxar2
/*     83 */	fmuld,s	%f244,%f212,%f212
/*     83 */	fmuld,s	%f244,%f214,%f214


/*     83 */	sxar2
/*     83 */	fmuld,s	%f184,%f246,%f184
/*     83 */	fmuld,s	%f186,%f248,%f186


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f252,%f166,%f198,%f178
/*     83 */	fmaddd,s	%f250,%f164,%f196,%f176


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f252,%f170,%f150,%f182
/*     83 */	fmaddd,s	%f250,%f168,%f148,%f180


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f250,%f172,%f220,%f250
/*     83 */	fmaddd,s	%f252,%f174,%f222,%f252


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f186,%f52,%f154,%f154
/*     83 */	fmaddd,s	%f184,%f50,%f152,%f152


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f186,%f56,%f158,%f158
/*     83 */	fmaddd,s	%f184,%f54,%f156,%f156


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f214,%f52,%f178,%f178
/*     83 */	fmaddd,s	%f212,%f50,%f176,%f176


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f214,%f56,%f182,%f182
/*     83 */	fmaddd,s	%f212,%f54,%f180,%f180


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f212,%f58,%f250,%f212
/*     83 */	fmaddd,s	%f214,%f60,%f252,%f214


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f186,%f60,%f162,%f162
/*     83 */	fmaddd,s	%f184,%f58,%f160,%f160


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f186,%f166,%f72,%f166
/*     83 */	fmaddd,s	%f184,%f164,%f68,%f164


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f186,%f170,%f78,%f170
/*     83 */	fmaddd,s	%f184,%f168,%f74,%f168


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f186,%f174,%f234,%f174
/*     83 */	fmaddd,s	%f184,%f172,%f232,%f172


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f186,%f178,%f120,%f178
/*     83 */	fmaddd,s	%f184,%f176,%f116,%f176


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f186,%f182,%f94,%f182
/*     83 */	fmaddd,s	%f184,%f180,%f90,%f180


/*     83 */	sxar2
/*     83 */	fmaddd,s	%f186,%f214,%f230,%f186
/*     83 */	fmaddd,s	%f184,%f212,%f96,%f184

.L5900:


.L5899:


.L5902:

/*     84 */	sxar1
/*     84 */	srl	%xg6,31,%g2

/* #00002 */	sethi	%hi(5136),%g5

/*     84 */	sxar1
/*     84 */	add	%g2,%xg6,%g2

/* #00002 */	xor	%g5,-17,%g5

/*     84 */	sra	%g2,1,%g2

/*    136 */	sxar1
/* #00002 */	ldd,s	[%fp+%g5],%f130

/* #00002 */	sethi	%hi(5152),%o0

/*     84 */	sra	%g2,%g0,%g2

/* #00002 */	xor	%o0,-33,%o0

/*     84 */	sllx	%g2,2,%g3

/*     38 */	sxar1
/* #00002 */	ldd,s	[%fp+%o0],%f132

/* #00002 */	sethi	%hi(5168),%o1

/*     84 */	add	%g3,%g2,%g3

/* #00002 */	xor	%o1,-49,%o1

/*     84 */	sllx	%g3,5,%g3

/*    153 */	sxar1
/* #00002 */	ldd,s	[%fp+%o1],%f134

/* #00002 */	sethi	%hi(5184),%o2


/*    136 */	sxar2
/*    136 */	ldd,s	[%xg11+%g3],%f250
/*    136 */	ldd,s	[%xg13+%g3],%f254

/* #00002 */	xor	%o2,-65,%o2


/*    136 */	sxar2
/*    136 */	add	%xg6,2,%xg6
/*    136 */	ldd,s	[%xg14+%g3],%f36


/*    195 */	sxar2
/*    195 */	ldd,s	[%xg16+%g3],%f48
/*    195 */	subcc	%xg8,1,%xg8


/*    136 */	sxar2
/*    136 */	ldd,s	[%xg15+%g3],%f42
/*    136 */	ldd,s	[%xg19+%g3],%f60


/*    136 */	sxar2
/*    136 */	ldd,s	[%xg17+%g3],%f52
/*    136 */	ldd,s	[%xg18+%g3],%f56


/*    177 */	sxar2
/*    177 */	fmsubd,sc	%f250,%f130,%f34,%f248
/*    177 */	fmsubd,sc	%f506,%f130,%f34,%f250


/*    136 */	sxar2
/*    136 */	ldd,s	[%xg20+%g3],%f72
/*    136 */	fmsubd,sc	%f254,%f130,%f40,%f252


/*    153 */	sxar2
/*    153 */	fmsubd,sc	%f510,%f130,%f40,%f254
/*    153 */	ldd,s	[%xg12+%g3],%f108


/*    177 */	sxar2
/*    177 */	fmsubd,sc	%f36,%f130,%f46,%f32
/*    177 */	fmsubd,sc	%f292,%f130,%f46,%f36


/*    136 */	sxar2
/* #00002 */	ldd,s	[%fp+%o2],%f136
/*    136 */	fmsubd,sc	%f42,%f130,%f70,%f38


/*    177 */	sxar2
/*    177 */	fmsubd,sc	%f48,%f130,%f76,%f44
/*    177 */	fmsubd,sc	%f304,%f130,%f76,%f48


/*    177 */	sxar2
/*    177 */	fmsubd,sc	%f60,%f130,%f92,%f58
/*    177 */	fmsubd,sc	%f298,%f130,%f70,%f42


/*     44 */	sxar2
/*     44 */	fmsubd,sc	%f52,%f130,%f86,%f50
/*     44 */	fmaddd,s	%f248,%f248,%f64,%f62


/*    136 */	sxar2
/*    136 */	fmaddd,s	%f250,%f250,%f64,%f66
/*    136 */	fmsubd,sc	%f56,%f130,%f118,%f54


/*    177 */	sxar2
/*    177 */	fmsubd,sc	%f316,%f130,%f92,%f60
/*    177 */	fmsubd,sc	%f308,%f130,%f86,%f52


/*    136 */	sxar2
/*    136 */	fmsubd,sc	%f312,%f130,%f118,%f56
/*    136 */	fmsubd,sc	%f72,%f130,%f140,%f68


/*     44 */	sxar2
/*     44 */	fmuld,s	%f252,%f44,%f74
/*     44 */	fmuld,s	%f254,%f48,%f78


/*     44 */	sxar2
/*     44 */	fmuld,s	%f44,%f44,%f80
/*     44 */	fmuld,s	%f48,%f48,%f82


/*     44 */	sxar2
/*     44 */	fmuld,s	%f252,%f58,%f84
/*     44 */	fmaddd,s	%f252,%f252,%f62,%f62


/*    177 */	sxar2
/*    177 */	fmaddd,s	%f254,%f254,%f66,%f66
/*    177 */	fmsubd,sc	%f328,%f130,%f140,%f72


/*     44 */	sxar2
/*     44 */	fmuld,s	%f254,%f60,%f88
/*     44 */	fmaddd,s	%f248,%f38,%f74,%f74


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f250,%f42,%f78,%f78
/*     44 */	fmaddd,s	%f38,%f38,%f80,%f80


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f42,%f42,%f82,%f82
/*     44 */	fmaddd,s	%f248,%f54,%f84,%f84


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f32,%f32,%f62,%f62
/*     44 */	fmaddd,s	%f36,%f36,%f66,%f66


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f250,%f56,%f88,%f88
/*     44 */	fmaddd,s	%f32,%f50,%f74,%f74


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f36,%f52,%f78,%f78
/*     44 */	fmaddd,s	%f50,%f50,%f80,%f80


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f52,%f52,%f82,%f82
/*     44 */	fmaddd,s	%f32,%f68,%f84,%f84


/*     38 */	sxar2
/*     38 */	frsqrtad,s	%f62,%f90
/*     38 */	fmuld,s	%f62,%f132,%f62


/*     38 */	sxar2
/*     38 */	frsqrtad,s	%f66,%f96
/*     38 */	fmuld,s	%f66,%f132,%f66


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f36,%f72,%f88,%f88
/*     44 */	faddd,s	%f80,%f84,%f80


/*     32 */	sxar2
/*     32 */	fmuld,s	%f90,%f90,%f94
/*     32 */	fmuld,s	%f96,%f96,%f98


/*     32 */	sxar2
/*     32 */	faddd,s	%f82,%f88,%f82
/*     32 */	fnmsubd,s	%f62,%f94,%f132,%f94


/*     32 */	sxar2
/*     32 */	fnmsubd,s	%f66,%f98,%f132,%f98
/*     32 */	fmaddd,s	%f90,%f94,%f90,%f90


/*     32 */	sxar2
/*     32 */	fmaddd,s	%f96,%f98,%f96,%f96
/*     32 */	fmuld,s	%f90,%f90,%f100


/*     32 */	sxar2
/*     32 */	fmuld,s	%f96,%f96,%f104
/*     32 */	fnmsubd,s	%f62,%f100,%f132,%f100


/*     32 */	sxar2
/*     32 */	fnmsubd,s	%f66,%f104,%f132,%f104
/*     32 */	fmaddd,s	%f90,%f100,%f90,%f90


/*     32 */	sxar2
/*     32 */	fmaddd,s	%f96,%f104,%f96,%f96
/*     32 */	fmuld,s	%f90,%f90,%f102


/*     32 */	sxar2
/*     32 */	fmuld,s	%f96,%f96,%f110
/*     32 */	fnmsubd,s	%f62,%f102,%f132,%f62


/*     32 */	sxar2
/*     32 */	fnmsubd,s	%f66,%f110,%f132,%f66
/*     32 */	fmaddd,s	%f90,%f62,%f90,%f90


/*     54 */	sxar2
/*     54 */	fmaddd,s	%f96,%f66,%f96,%f96
/*     54 */	fmuld,s	%f90,%f90,%f106


/*     54 */	sxar2
/*     54 */	fmaddd,sc	%f108,%f90,%f134,%f90
/*     54 */	fmuld,s	%f96,%f96,%f112


/*     54 */	sxar2
/*     54 */	fmaddd,sc	%f364,%f96,%f134,%f108
/*     54 */	fmuld,s	%f74,%f106,%f74


/*     54 */	sxar2
/*     54 */	fmuld,s	%f90,%f106,%f90
/*     54 */	fmuld,s	%f108,%f112,%f108


/*     44 */	sxar2
/*     44 */	fmuld,s	%f78,%f112,%f78
/*     44 */	fmuld,s	%f106,%f80,%f106


/*     54 */	sxar2
/*     54 */	fmuld,s	%f112,%f82,%f112
/*     54 */	fmuld,s	%f136,%f74,%f114


/*     54 */	sxar2
/*     54 */	fmaddd,s	%f90,%f248,%f152,%f152
/*     54 */	fmuld,s	%f136,%f78,%f116


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f108,%f250,%f154,%f154
/*     44 */	fmaddd,s	%f74,%f74,%f106,%f74


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f78,%f78,%f112,%f78
/*     44 */	fmaddd,s	%f108,%f254,%f158,%f158


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f90,%f252,%f156,%f156
/*     44 */	fmaddd,s	%f108,%f36,%f162,%f162


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f90,%f32,%f160,%f160
/*     44 */	fmaddd,s	%f114,%f248,%f38,%f38


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f114,%f252,%f44,%f44
/*     44 */	fmaddd,s	%f116,%f250,%f42,%f42


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f116,%f254,%f48,%f48
/*     44 */	faddd,s	%f114,%f114,%f120


/*     54 */	sxar2
/*     54 */	faddd,s	%f116,%f116,%f122
/*     54 */	fmuld,s	%f136,%f74,%f74


/*     44 */	sxar2
/*     44 */	fmuld,s	%f136,%f78,%f78
/*     44 */	fmaddd,s	%f114,%f32,%f50,%f114


/*     24 */	sxar2
/*     24 */	fmaddd,s	%f116,%f36,%f52,%f116
/*     24 */	fmaddd,s	%f90,%f38,%f164,%f164


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f90,%f44,%f168,%f168
/*     44 */	fmaddd,s	%f108,%f42,%f166,%f166


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f108,%f48,%f170,%f170
/*     44 */	fmaddd,s	%f122,%f42,%f56,%f42


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f120,%f38,%f54,%f38
/*     44 */	fmaddd,s	%f122,%f48,%f60,%f48


/*     54 */	sxar2
/*     54 */	fmaddd,s	%f120,%f44,%f58,%f44
/*     54 */	fmaddd,s	%f120,%f114,%f68,%f120


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f122,%f116,%f72,%f122
/*     44 */	fmaddd,s	%f108,%f116,%f174,%f174


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f90,%f114,%f172,%f172
/*     44 */	fmaddd,s	%f78,%f250,%f42,%f250


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f74,%f248,%f38,%f248
/*     44 */	fmaddd,s	%f78,%f254,%f48,%f254


/*     54 */	sxar2
/*     54 */	fmaddd,s	%f74,%f252,%f44,%f252
/*     54 */	fmaddd,s	%f74,%f32,%f120,%f74


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f78,%f36,%f122,%f78
/*     44 */	fmaddd,s	%f108,%f250,%f178,%f178


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f90,%f248,%f176,%f176
/*     44 */	fmaddd,s	%f108,%f254,%f182,%f182


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f90,%f252,%f180,%f180
/*     44 */	fmaddd,s	%f108,%f78,%f186,%f186

/*     24 */	sxar1
/*     24 */	fmaddd,s	%f90,%f74,%f184,%f184

/*    195 */	bne,pt	%icc, .L5902
	nop


.L5898:


/*    195 */	sxar2
/*    195 */	std,s	%f152,[%fp+-2625]
/*    195 */	std,s	%f154,[%fp+-2609]


/*    195 */	sxar2
/*    195 */	std,s	%f156,[%fp+-2593]
/*    195 */	std,s	%f158,[%fp+-2577]


/*    195 */	sxar2
/*    195 */	std,s	%f160,[%fp+-2561]
/*    195 */	std,s	%f162,[%fp+-2545]


/*    195 */	sxar2
/*    195 */	std,s	%f164,[%fp+-2529]
/*    195 */	std,s	%f166,[%fp+-2513]


/*    195 */	sxar2
/*    195 */	std,s	%f168,[%fp+-2497]
/*    195 */	std,s	%f170,[%fp+-2481]


/*    195 */	sxar2
/*    195 */	std,s	%f172,[%fp+-2465]
/*    195 */	std,s	%f174,[%fp+-2449]


/*    195 */	sxar2
/*    195 */	std,s	%f176,[%fp+-2433]
/*    195 */	std,s	%f178,[%fp+-2417]


/*    195 */	sxar2
/*    195 */	std,s	%f180,[%fp+-2401]
/*    195 */	std,s	%f182,[%fp+-2385]


/*    195 */	sxar2
/*    195 */	std,s	%f184,[%fp+-2369]
/*    195 */	std,s	%f186,[%fp+-2353]

.L5752:

/*    136 *//*    136 */	call	__mpc_obar
/*    136 */	ldx	[%fp+2199],%o0

/*    136 */

/*     88 */	sxar2
/*     88 */	add	%l7,%l4,%xg26
/*     88 */	ldd,s	[%fp+-2625],%f34
/*     88 */	sxar1
/*     88 */	ldd,s	[%fp+-2609],%f32

/*    141 */	add	%l2,2,%l2

/*     88 */	sxar2
/*     88 */	ldd,s	[%fp+-2593],%f38
/*     88 */	ldd,s	[%fp+-2577],%f36


/*     88 */	sxar2
/*     88 */	ldd,s	[%fp+-2561],%f42
/*     88 */	ldd,s	[%fp+-2545],%f40
/*     88 */	sxar2
/*     88 */	ldd,s	[%fp+-2529],%f46
/*     88 */	ldd,s	[%fp+-2513],%f44
/*     88 */	sxar2
/*     88 */	faddd,s	%f34,%f32,%f34
/*     88 */	ldd,s	[%fp+-2497],%f50
/*     88 */	sxar2
/*     88 */	ldd,s	[%fp+-2481],%f48
/*     88 */	faddd,s	%f38,%f36,%f38
/*     88 */	sxar2
/*     88 */	ldd,s	[%fp+-2465],%f54
/*     88 */	ldd,s	[%fp+-2449],%f52
/*     88 */	sxar2
/*     88 */	faddd,s	%f42,%f40,%f42
/*     88 */	ldd,s	[%fp+-2433],%f58
/*     88 */	sxar2
/*     88 */	ldd,s	[%fp+-2417],%f56
/*     88 */	faddd,s	%f46,%f44,%f46
/*     88 */	sxar2
/*     88 */	ldd,s	[%fp+-2401],%f62
/*     88 */	ldd,s	[%fp+-2385],%f60
/*     88 */	sxar2
/*     88 */	faddd,s	%f50,%f48,%f50
/*     88 */	ldd,s	[%fp+-2369],%f66
/*     88 */	sxar2
/*     88 */	ldd,s	[%fp+-2353],%f64
/*     88 */	faddd,s	%f54,%f52,%f54
/*     88 */	sxar2
/*     88 */	faddd,s	%f58,%f56,%f58
/*     88 */	faddd,s	%f62,%f60,%f62

/*     21 */	sxar2
/*     21 */	faddd,s	%f66,%f64,%f66
/*     21 */	std,s	%f34,[%xg26]
/*     21 */	sxar2
/*     21 */	std,s	%f38,[%xg26+16]
/*     21 */	std,s	%f42,[%xg26+32]
/*     21 */	sxar2
/*     21 */	std,s	%f46,[%xg26+48]
/*     21 */	std,s	%f50,[%xg26+64]
/*     21 */	sxar2
/*     21 */	std,s	%f54,[%xg26+80]
/*     21 */	std,s	%f58,[%xg26+96]
/*     21 */	sxar2
/*     21 */	std,s	%f62,[%xg26+112]
/*     21 */	std,s	%f66,[%xg26+128]
/*     21 */	ldsw	[%i0+2195],%g4
/*     21 */	cmp	%l2,%g4
/*     21 */	bl,pt	%icc, .L5739
/*     21 */	add	%l4,144,%l4


.L5753:


.L5754:


/*    142 */	call	__mpc_obar
/*    142 */	ldx	[%fp+2199],%o0


.L5755:

/*    142 */	ret
	restore



.LLFE7:
	.size	_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_2,.-_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_2
	.type	_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_2,#function
	.section	".gcc_except_table",#alloc
	.align	8
.LLLSDA7:
	.byte	255
	.byte	255
	.byte	1
	.uleb128	.LLLSDACSE7-.LLLSDACSB7
.LLLSDACSB7:
	.uleb128	.LLEHB3-.LLFB7
	.uleb128	.LLEHE3-.LLEHB3
	.uleb128	.L5733-.LLFB7
	.uleb128	0x0
.LLLSDACSE7:
	.sleb128	0
	.sleb128	0
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite6-k.cpp _ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_3 $"
	.section	".text"
	.align	64
_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_3:
.LLFB8:
.L5757:

/*    144 */	save	%sp,-448,%sp
.LLCFI6:
/*    144 */	stx	%i0,[%fp+2175]
/*    144 */	stx	%i3,[%fp+2199]
/*    144 */	stx	%i0,[%fp+2175]

.L5758:

/*    144 *//*    144 */	sxar1
/*    144 */	ldsw	[%i0+2035],%xg20
/*    144 */
/*    144 */
/*    144 */
/*    145 */	ldsw	[%i0+2187],%o0
/*    145 */	ldsw	[%i0+2195],%l1
/*    145 */	cmp	%o0,%l1
/*    145 */	bge	.L5771
	nop


.L5759:

/*    145 */	sxar1
/*    145 */	mov	1,%xg19

/*    145 */	sra	%l1,%g0,%l1


/*    145 */	sxar2
/*    145 */	fzero,s	%f42
/*    145 */	stx	%xg19,[%fp+2031]

/*    145 */	sxar1
/*    ??? */	std,s	%f42,[%fp+1823]

.L5760:

/*    145 */	sethi	%h44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%l2

/*    145 */	mov	1,%l7

/*    145 */	or	%l2,%m44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%l2

/*    145 */	sethi	%hi(77760),%i1

/*    145 */	sllx	%l2,12,%l2

/*    145 */	sethi	%hi(151488),%i2

/*    145 */	or	%l2,%l44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%l2

/*    145 */	sethi	%hi(225216),%i3

/*    145 */	add	%fp,2039,%l3

/*    145 */	add	%fp,2023,%l4

/*    145 */	add	%fp,2031,%l5

/*    145 */	sra	%l7,%g0,%l6

/*    145 */	or	%i1,960,%i1

/*    145 */	or	%i2,960,%i2

/*    145 */	or	%i3,960,%i3

/*    145 */	sethi	%hi(294912),%l0

/*    145 */	sethi	%hi(73728),%i4

.L5761:

/*    145 */	sra	%o0,%g0,%o0

/*    145 */	stx	%g0,[%sp+2223]

/*    145 */	mov	2,%o2

/*    145 */	mov	%g0,%o3

/*    145 */	mov	%l1,%o1

/*    145 */	mov	%l3,%o4


/*    145 */	stx	%g0,[%sp+2231]

/*    145 */	stx	%l5,[%sp+2239]


/*    145 */	sxar2
/*    145 */	ldx	[%fp+2199],%xg17
/*    145 */	stx	%xg17,[%sp+2247]

/*    145 */	call	__mpc_ostd_th
/*    145 */	mov	%l4,%o5
/*    145 */	sxar2
/*    145 */	ldx	[%fp+2031],%xg18
/*    145 */	cmp	%xg18,%g0
/*    145 */	ble,pn	%xcc, .L5771
	nop


.L5762:

/*    145 */	ldx	[%fp+2039],%o0


/*    145 */	sxar2
/*    145 */	ldx	[%fp+2023],%xg0
/*    145 */	ldsw	[%i0+2187],%xg7


/*    145 */	sxar2
/*    145 */	ldx	[%i0+2207],%xg8
/*    145 */	ldsw	[%i0+2027],%xg11

/*    145 */	sra	%o0,%g0,%o0


/*    145 */	sxar2
/*    145 */	sra	%xg0,%g0,%xg0
/*    145 */	sub	%xg0,%o0,%xg0


/*    145 */	sxar2
/*    145 */	add	%o0,1,%xg1
/*    145 */	srl	%xg0,31,%xg2


/*    145 */	sxar2
/*    145 */	sra	%o0,%g0,%xg3
/*    145 */	add	%xg0,%xg2,%xg0


/*    145 */	sxar2
/*    145 */	sra	%xg1,%g0,%xg1
/*    145 */	sra	%xg0,1,%xg0


/*    145 */	sxar2
/*    145 */	sllx	%xg3,3,%xg4
/*    145 */	add	%xg0,1,%xg0


/*    145 */	sxar2
/*    145 */	sllx	%xg1,3,%xg5
/*    145 */	sra	%xg0,%g0,%xg0


/*    145 */	sxar2
/*    145 */	add	%xg4,%xg3,%xg4
/*    145 */	sub	%l6,%xg0,%xg0


/*    145 */	sxar2
/*    145 */	add	%xg5,%xg1,%xg5
/*    145 */	srax	%xg0,32,%xg6


/*    145 */	sxar2
/*    145 */	sllx	%xg4,3,%xg4
/*    145 */	and	%xg0,%xg6,%xg0


/*    145 */	sxar2
/*    145 */	sllx	%xg5,3,%xg5
/*    145 */	sub	%l7,%xg0,%xg0


/*    145 */	sxar2
/*    145 */	add	%xg8,%xg4,%xg4
/*    145 */	sub	%o0,%xg7,%xg7

/*    145 */	sxar1
/*    145 */	add	%xg8,%xg5,%xg8

.L5763:


/*     25 */	sxar2
/*     25 */	srl	%xg7,31,%xg9
/*    ??? */	ldd,s	[%fp+1823],%f40


/*    146 */	sxar2
/*    146 */	add	%xg9,%xg7,%xg9
/*    146 */	sra	%xg9,1,%xg9


/*     25 */	sxar2
/*     25 */	std,s	%f40,[%fp+1839]
/*     25 */	std,s	%f40,[%fp+1855]


/*     25 */	sxar2
/*     25 */	std,s	%f40,[%fp+1871]
/*     25 */	std,s	%f40,[%fp+1887]


/*     25 */	sxar2
/*     25 */	std,s	%f40,[%fp+1903]
/*     25 */	std,s	%f40,[%fp+1919]


/*     25 */	sxar2
/*     25 */	std,s	%f40,[%fp+1935]
/*     25 */	std,s	%f40,[%fp+1951]

/*     25 */	sxar1
/*     25 */	std,s	%f40,[%fp+1967]

.L5764:

/*    150 */	sxar1
/*    150 */	cmp	%xg11,%g0

/*    150 */	ble	.L5768
	nop


.L5765:


/*    150 */	sxar2
/*    150 */	sra	%xg9,%g0,%xg9
/*    ??? */	ldd,s	[%fp+1823],%f64


/*    150 */	sxar2
/*    150 */	sub	%xg11,4,%xg10
/*    150 */	sllx	%xg9,3,%xg12


/*    150 */	sxar2
/*    150 */	cmp	%xg10,%g0
/*    150 */	add	%xg12,%xg9,%xg12


/*    150 */	sxar2
/*    150 */	sllx	%xg12,4,%xg12
/*    150 */	fmovd,s	%f64,%f60


/*    150 */	sxar2
/*    150 */	fmovd,s	%f60,%f56
/*    150 */	fmovd,s	%f60,%f52


/*    150 */	sxar2
/*    150 */	fmovd,s	%f60,%f48
/*    150 */	fmovd,s	%f60,%f44


/*    150 */	sxar2
/*    150 */	fmovd,s	%f60,%f40
/*    150 */	fmovd,s	%f60,%f36

/*    150 */	sxar1
/*    150 */	fmovd,s	%f60,%f32

/*    150 */	bl	.L5774
	nop


.L5777:


/*    161 */	sxar2
/*    161 */	fzero,s	%f68
/*    161 */	add	%l2,%xg12,%xg13


/*    150 */	sxar2
/*    150 */	fmovd,s	%f60,%f56
/*    150 */	cmp	%xg10,16


/*    161 */	sxar2
/*    161 */	add	%i1,%xg13,%xg14
/*    161 */	add	%i2,%xg13,%xg15


/*    161 */	sxar2
/*    161 */	fmovd,s	%f60,%f52
/*    161 */	fmovd,s	%f60,%f48


/*    161 */	sxar2
/*    161 */	add	%i3,%xg13,%xg16
/*    161 */	fmovd,s	%f60,%f44


/*    161 */	sxar2
/*    161 */	fmovd,s	%f60,%f40
/*    161 */	fmovd,s	%f60,%f36


/*    150 */	sxar2
/*    150 */	fmovd,s	%f60,%f32
/*    150 */	fmovd,s	%f68,%f76


/*    161 */	sxar2
/*    161 */	fmovd,s	%f68,%f72
/*    161 */	fmovd,s	%f76,%f80


/*    161 */	sxar2
/*    161 */	fmovd,s	%f76,%f84
/*    161 */	fmovd,s	%f76,%f88


/*    161 */	sxar2
/*    161 */	fmovd,s	%f76,%f92
/*    161 */	fmovd,s	%f76,%f96


/*    161 */	sxar2
/*    161 */	fmovd,s	%f76,%f100
/*    161 */	fmovd,s	%f76,%f104


/*    161 */	sxar2
/*    161 */	fmovd,s	%f76,%f108
/*    161 */	fmovd,s	%f76,%f112


/*    161 */	sxar2
/*    161 */	fmovd,s	%f76,%f116
/*    161 */	fmovd,s	%f76,%f120


/*    161 */	sxar2
/*    161 */	fmovd,s	%f76,%f124
/*    161 */	fmovd,s	%f76,%f128


/*    161 */	sxar2
/*    161 */	fmovd,s	%f76,%f132
/*    161 */	fmovd,s	%f76,%f136


/*    161 */	sxar2
/*    161 */	fmovd,s	%f76,%f140
/*    161 */	fmovd,s	%f76,%f144


/*    161 */	sxar2
/*    161 */	fmovd,s	%f76,%f148
/*    161 */	fmovd,s	%f76,%f152


/*    161 */	sxar2
/*    161 */	fmovd,s	%f76,%f156
/*    161 */	fmovd,s	%f76,%f160


/*    161 */	sxar2
/*    161 */	fmovd,s	%f76,%f164
/*    161 */	fmovd,s	%f76,%f168

/*    161 */	sxar1
/*    161 */	fmovd,s	%f76,%f172

/*    150 */	bl	.L5909
	nop


.L5905:


.L5912:


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg13],%f34
/*    150 */	ldd,s	[%xg13+32],%f42


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg13+16],%f38
/*    150 */	ldd,s	[%xg13+48],%f46


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg13+80],%f54
/*    150 */	ldd,s	[%xg13+64],%f50

.L5766:


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg13+96],%f176
/*    150 */	faddd,s	%f32,%f34,%f32


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg13+112],%f178
/*    150 */	faddd,s	%f36,%f38,%f36


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg13+128],%f180
/*    150 */	faddd,s	%f40,%f42,%f40


/*    150 */	sxar2
/*    150 */	faddd,s	%f44,%f46,%f44
/*    150 */	ldd,s	[%xg14+-4032],%f182


/*    150 */	sxar2
/*    150 */	faddd,s	%f48,%f50,%f48
/*    150 */	faddd,s	%f52,%f54,%f52


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg14+-4016],%f184
/*    150 */	faddd,s	%f56,%f176,%f56


/*    150 */	sxar2
/*    150 */	faddd,s	%f60,%f178,%f60
/*    150 */	ldd,s	[%xg14+-4000],%f186


/*    150 */	sxar2
/*    150 */	faddd,s	%f64,%f180,%f64
/*    150 */	faddd,s	%f68,%f182,%f68


/*    150 */	sxar2
/*    150 */	faddd,s	%f72,%f184,%f72
/*    150 */	ldd,s	[%xg14+-3984],%f188


/*    150 */	sxar2
/*    150 */	faddd,s	%f76,%f186,%f76
/*    150 */	ldd,s	[%xg14+-3968],%f190


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg14+-3952],%f192
/*    150 */	ldd,s	[%xg14+-3936],%f194


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg14+-3920],%f196
/*    150 */	ldd,s	[%xg14+-3904],%f198


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg15+-4032],%f200
/*    150 */	ldd,s	[%xg15+-4016],%f202


/*    150 */	sxar2
/*    150 */	faddd,s	%f80,%f188,%f80
/*    150 */	ldd,s	[%xg15+-4000],%f204


/*    150 */	sxar2
/*    150 */	faddd,s	%f84,%f190,%f84
/*    150 */	ldd,s	[%xg15+-3984],%f206


/*    150 */	sxar2
/*    150 */	faddd,s	%f88,%f192,%f88
/*    150 */	ldd,s	[%xg15+-3968],%f208


/*    150 */	sxar2
/*    150 */	faddd,s	%f92,%f194,%f92
/*    150 */	faddd,s	%f96,%f196,%f96


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg15+-3952],%f210
/*    150 */	faddd,s	%f100,%f198,%f100


/*    150 */	sxar2
/*    150 */	faddd,s	%f104,%f200,%f104
/*    150 */	faddd,s	%f108,%f202,%f108


/*    150 */	sxar2
/*    150 */	faddd,s	%f112,%f204,%f112
/*    150 */	faddd,s	%f116,%f206,%f116


/*    150 */	sxar2
/*    150 */	faddd,s	%f120,%f208,%f120
/*    150 */	faddd,s	%f124,%f210,%f124


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg15+-3936],%f212
/*    150 */	ldd,s	[%xg15+-3920],%f214


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg15+-3904],%f216
/*    150 */	ldd,s	[%xg16+-4032],%f218


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg16+-4016],%f220
/*    150 */	ldd,s	[%xg16+-4000],%f222


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg16+-3984],%f224
/*    150 */	ldd,s	[%xg16+-3968],%f226


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg16+-3952],%f228
/*    150 */	faddd,s	%f128,%f212,%f128


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg16+-3936],%f230
/*    150 */	ldd,s	[%xg16+-3920],%f232


/*    150 */	sxar2
/*    150 */	faddd,s	%f132,%f214,%f132
/*    150 */	ldd,s	[%xg16+-3904],%f234


/*    150 */	sxar2
/*    150 */	faddd,s	%f136,%f216,%f136
/*    150 */	faddd,s	%f140,%f218,%f140


/*    150 */	sxar2
/*    150 */	faddd,s	%f144,%f220,%f144
/*    150 */	faddd,s	%f148,%f222,%f148


/*    150 */	sxar2
/*    150 */	faddd,s	%f152,%f224,%f152
/*    150 */	faddd,s	%f156,%f226,%f156


/*    150 */	sxar2
/*    150 */	faddd,s	%f160,%f228,%f160
/*    150 */	faddd,s	%f164,%f230,%f164


/*    150 */	sxar2
/*    150 */	add	%l0,%xg13,%xg13
/*    150 */	faddd,s	%f168,%f232,%f168


/*    150 */	sxar2
/*    150 */	faddd,s	%f172,%f234,%f172
/*    150 */	ldd,s	[%xg13],%f236


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg13+32],%f240
/*    150 */	ldd,s	[%xg13+16],%f238


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg13+48],%f242
/*    150 */	ldd,s	[%xg13+80],%f246


/*    150 */	sxar2
/*    150 */	add	%xg12,%l0,%xg12
/*    150 */	ldd,s	[%xg13+64],%f244


/*    150 */	sxar2
/*    150 */	add	%l0,%xg16,%xg16
/*    150 */	add	%l0,%xg15,%xg15


/*    150 */	sxar2
/*    150 */	add	%l0,%xg14,%xg14
/*    150 */	ldd,s	[%xg13+96],%f248


/*    150 */	sxar2
/*    150 */	faddd,s	%f32,%f236,%f32
/*    150 */	ldd,s	[%xg13+112],%f250


/*    150 */	sxar2
/*    150 */	faddd,s	%f36,%f238,%f36
/*    150 */	ldd,s	[%xg13+128],%f252


/*    150 */	sxar2
/*    150 */	faddd,s	%f40,%f240,%f40
/*    150 */	faddd,s	%f44,%f242,%f44


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg14+-4032],%f254
/*    150 */	faddd,s	%f48,%f244,%f48


/*    150 */	sxar2
/*    150 */	faddd,s	%f52,%f246,%f52
/*    150 */	ldd,s	[%xg14+-4016],%f34


/*    150 */	sxar2
/*    150 */	faddd,s	%f56,%f248,%f56
/*    150 */	faddd,s	%f60,%f250,%f60


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg14+-4000],%f38
/*    150 */	faddd,s	%f64,%f252,%f64


/*    150 */	sxar2
/*    150 */	faddd,s	%f68,%f254,%f68
/*    150 */	faddd,s	%f72,%f34,%f72


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg14+-3984],%f42
/*    150 */	faddd,s	%f76,%f38,%f76


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg14+-3968],%f46
/*    150 */	ldd,s	[%xg14+-3952],%f50


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg14+-3936],%f54
/*    150 */	ldd,s	[%xg14+-3920],%f58


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg14+-3904],%f62
/*    150 */	ldd,s	[%xg15+-4032],%f66


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg15+-4016],%f70
/*    150 */	faddd,s	%f80,%f42,%f80


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg15+-4000],%f74
/*    150 */	faddd,s	%f84,%f46,%f84


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg15+-3984],%f78
/*    150 */	faddd,s	%f88,%f50,%f88


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg15+-3968],%f82
/*    150 */	faddd,s	%f92,%f54,%f92


/*    150 */	sxar2
/*    150 */	faddd,s	%f96,%f58,%f96
/*    150 */	ldd,s	[%xg15+-3952],%f86


/*    150 */	sxar2
/*    150 */	faddd,s	%f100,%f62,%f100
/*    150 */	faddd,s	%f104,%f66,%f104


/*    150 */	sxar2
/*    150 */	faddd,s	%f108,%f70,%f108
/*    150 */	faddd,s	%f112,%f74,%f112


/*    150 */	sxar2
/*    150 */	faddd,s	%f116,%f78,%f116
/*    150 */	faddd,s	%f120,%f82,%f120


/*    150 */	sxar2
/*    150 */	faddd,s	%f124,%f86,%f124
/*    150 */	ldd,s	[%xg15+-3936],%f90


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg15+-3920],%f94
/*    150 */	ldd,s	[%xg15+-3904],%f98


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg16+-4032],%f102
/*    150 */	ldd,s	[%xg16+-4016],%f106


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg16+-4000],%f110
/*    150 */	ldd,s	[%xg16+-3984],%f114


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg16+-3968],%f118
/*    150 */	ldd,s	[%xg16+-3952],%f122


/*    150 */	sxar2
/*    150 */	faddd,s	%f128,%f90,%f128
/*    150 */	ldd,s	[%xg16+-3936],%f126


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg16+-3920],%f130
/*    150 */	faddd,s	%f132,%f94,%f132


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg16+-3904],%f134
/*    150 */	faddd,s	%f136,%f98,%f136


/*    150 */	sxar2
/*    150 */	faddd,s	%f140,%f102,%f140
/*    150 */	faddd,s	%f144,%f106,%f144


/*    150 */	sxar2
/*    150 */	faddd,s	%f148,%f110,%f148
/*    150 */	faddd,s	%f152,%f114,%f152


/*    150 */	sxar2
/*    150 */	faddd,s	%f156,%f118,%f156
/*    150 */	faddd,s	%f160,%f122,%f160


/*    150 */	sxar2
/*    150 */	faddd,s	%f164,%f126,%f164
/*    150 */	add	%l0,%xg13,%xg13


/*    150 */	sxar2
/*    150 */	faddd,s	%f168,%f130,%f168
/*    150 */	faddd,s	%f172,%f134,%f172


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg13],%f34
/*    150 */	ldd,s	[%xg13+32],%f42


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg13+16],%f38
/*    150 */	ldd,s	[%xg13+48],%f46


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg13+80],%f54
/*    150 */	add	%xg12,%l0,%xg12


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg13+64],%f50
/*    150 */	add	%l0,%xg16,%xg16


/*    150 */	sxar2
/*    150 */	add	%l0,%xg15,%xg15
/*    150 */	add	%l0,%xg14,%xg14


/*    150 */	sxar2
/*    150 */	sub	%xg10,8,%xg10
/*    150 */	cmp	%xg10,19

/*    150 */	bge,pt	%icc, .L5766
	nop


.L5913:


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg13+96],%f58
/*    150 */	faddd,s	%f32,%f34,%f32


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg13+112],%f62
/*    150 */	faddd,s	%f36,%f38,%f36


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg13+128],%f66
/*    150 */	faddd,s	%f40,%f42,%f40


/*    150 */	sxar2
/*    150 */	faddd,s	%f44,%f46,%f44
/*    150 */	ldd,s	[%xg14+-4032],%f70


/*    150 */	sxar2
/*    150 */	faddd,s	%f48,%f50,%f48
/*    150 */	faddd,s	%f52,%f54,%f52


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg14+-4016],%f74
/*    150 */	ldd,s	[%xg14+-4000],%f78


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg14+-3984],%f82
/*    150 */	ldd,s	[%xg14+-3968],%f86


/*    150 */	sxar2
/*    150 */	add	%l0,%xg13,%xg13
/*    150 */	add	%xg12,%l0,%xg12


/*    150 */	sxar2
/*    150 */	faddd,s	%f56,%f58,%f56
/*    150 */	faddd,s	%f60,%f62,%f60


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg14+-3952],%f90
/*    150 */	ldd,s	[%xg14+-3936],%f94


/*    150 */	sxar2
/*    150 */	faddd,s	%f64,%f66,%f64
/*    150 */	faddd,s	%f68,%f70,%f68


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg14+-3920],%f98
/*    150 */	ldd,s	[%xg14+-3904],%f102


/*    150 */	sxar2
/*    150 */	faddd,s	%f72,%f74,%f72
/*    150 */	faddd,s	%f76,%f78,%f76


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg15+-4032],%f106
/*    150 */	ldd,s	[%xg15+-4016],%f110


/*    150 */	sxar2
/*    150 */	faddd,s	%f80,%f82,%f80
/*    150 */	ldd,s	[%xg15+-4000],%f114


/*    150 */	sxar2
/*    150 */	faddd,s	%f84,%f86,%f84
/*    150 */	ldd,s	[%xg15+-3984],%f118


/*    150 */	sxar2
/*    150 */	faddd,s	%f88,%f90,%f88
/*    150 */	ldd,s	[%xg15+-3968],%f122


/*    150 */	sxar2
/*    150 */	faddd,s	%f92,%f94,%f92
/*    150 */	ldd,s	[%xg15+-3952],%f126


/*    150 */	sxar2
/*    150 */	faddd,s	%f96,%f98,%f96
/*    150 */	faddd,s	%f100,%f102,%f100


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg15+-3936],%f130
/*    150 */	ldd,s	[%xg15+-3920],%f134


/*    150 */	sxar2
/*    150 */	faddd,s	%f104,%f106,%f104
/*    150 */	faddd,s	%f108,%f110,%f108


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg15+-3904],%f138
/*    150 */	ldd,s	[%xg16+-4032],%f142


/*    150 */	sxar2
/*    150 */	faddd,s	%f112,%f114,%f112
/*    150 */	faddd,s	%f116,%f118,%f116


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg16+-4016],%f146
/*    150 */	ldd,s	[%xg16+-4000],%f150


/*    150 */	sxar2
/*    150 */	faddd,s	%f120,%f122,%f120
/*    150 */	faddd,s	%f124,%f126,%f124


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg16+-3984],%f154
/*    150 */	ldd,s	[%xg16+-3968],%f158


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg16+-3952],%f162
/*    150 */	faddd,s	%f128,%f130,%f128


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg16+-3936],%f166
/*    150 */	faddd,s	%f132,%f134,%f132


/*    150 */	sxar2
/*    150 */	ldd,s	[%xg16+-3920],%f170
/*    150 */	ldd,s	[%xg16+-3904],%f174


/*    150 */	sxar2
/*    150 */	faddd,s	%f136,%f138,%f136
/*    150 */	faddd,s	%f140,%f142,%f140


/*    150 */	sxar2
/*    150 */	faddd,s	%f144,%f146,%f144
/*    150 */	faddd,s	%f148,%f150,%f148


/*    150 */	sxar2
/*    150 */	add	%l0,%xg16,%xg16
/*    150 */	add	%l0,%xg15,%xg15


/*    150 */	sxar2
/*    150 */	faddd,s	%f152,%f154,%f152
/*    150 */	faddd,s	%f156,%f158,%f156


/*    150 */	sxar2
/*    150 */	add	%l0,%xg14,%xg14
/*    150 */	sub	%xg10,4,%xg10


/*    150 */	sxar2
/*    150 */	faddd,s	%f160,%f162,%f160
/*    150 */	faddd,s	%f164,%f166,%f164


/*    150 */	sxar2
/*    150 */	faddd,s	%f168,%f170,%f168
/*    150 */	faddd,s	%f172,%f174,%f172

.L5909:


.L5908:


.L5911:


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg13],%f138
/*     43 */	ldd,s	[%xg13+16],%f142


/*    161 */	sxar2
/*    161 */	add	%xg12,%l0,%xg12
/*    161 */	subcc	%xg10,4,%xg10


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg13+32],%f146
/*     43 */	ldd,s	[%xg13+48],%f150


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg13+64],%f154
/*     43 */	ldd,s	[%xg13+80],%f158


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg13+96],%f162
/*     43 */	ldd,s	[%xg13+112],%f166


/*     43 */	sxar2
/*     43 */	faddd,s	%f32,%f138,%f32
/*     43 */	faddd,s	%f36,%f142,%f36


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg13+128],%f170
/*     43 */	ldd,s	[%xg14+-4032],%f174


/*     43 */	sxar2
/*     43 */	faddd,s	%f40,%f146,%f40
/*     43 */	faddd,s	%f44,%f150,%f44


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg14+-4016],%f176
/*     43 */	ldd,s	[%xg14+-4000],%f178


/*     43 */	sxar2
/*     43 */	faddd,s	%f48,%f154,%f48
/*     43 */	faddd,s	%f52,%f158,%f52


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg14+-3984],%f180
/*     43 */	ldd,s	[%xg14+-3968],%f182


/*     43 */	sxar2
/*     43 */	faddd,s	%f56,%f162,%f56
/*     43 */	faddd,s	%f60,%f166,%f60


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg14+-3952],%f184
/*     43 */	ldd,s	[%xg14+-3936],%f186


/*     43 */	sxar2
/*     43 */	faddd,s	%f64,%f170,%f64
/*     43 */	faddd,s	%f68,%f174,%f68


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg14+-3920],%f188
/*     43 */	ldd,s	[%xg14+-3904],%f190


/*     43 */	sxar2
/*     43 */	faddd,s	%f72,%f176,%f72
/*     43 */	faddd,s	%f76,%f178,%f76


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg15+-4032],%f192
/*     43 */	ldd,s	[%xg15+-4016],%f194


/*     43 */	sxar2
/*     43 */	faddd,s	%f80,%f180,%f80
/*     43 */	faddd,s	%f84,%f182,%f84


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg15+-4000],%f196
/*     43 */	ldd,s	[%xg15+-3984],%f198


/*     43 */	sxar2
/*     43 */	faddd,s	%f88,%f184,%f88
/*     43 */	ldd,s	[%xg15+-3968],%f200


/*     43 */	sxar2
/*     43 */	faddd,s	%f92,%f186,%f92
/*     43 */	ldd,s	[%xg15+-3952],%f202


/*     43 */	sxar2
/*     43 */	faddd,s	%f96,%f188,%f96
/*     43 */	ldd,s	[%xg15+-3936],%f204


/*     43 */	sxar2
/*     43 */	faddd,s	%f100,%f190,%f100
/*     43 */	ldd,s	[%xg15+-3920],%f206


/*     43 */	sxar2
/*     43 */	faddd,s	%f104,%f192,%f104
/*     43 */	ldd,s	[%xg15+-3904],%f208


/*     43 */	sxar2
/*     43 */	faddd,s	%f108,%f194,%f108
/*     43 */	ldd,s	[%xg16+-4032],%f210


/*     43 */	sxar2
/*     43 */	faddd,s	%f112,%f196,%f112
/*     43 */	ldd,s	[%xg16+-4016],%f212


/*     43 */	sxar2
/*     43 */	faddd,s	%f116,%f198,%f116
/*     43 */	ldd,s	[%xg16+-4000],%f214


/*     43 */	sxar2
/*     43 */	faddd,s	%f120,%f200,%f120
/*     43 */	faddd,s	%f124,%f202,%f124


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg16+-3984],%f216
/*     43 */	ldd,s	[%xg16+-3968],%f218


/*     43 */	sxar2
/*     43 */	faddd,s	%f128,%f204,%f128
/*     43 */	faddd,s	%f132,%f206,%f132


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg16+-3952],%f220
/*     43 */	ldd,s	[%xg16+-3936],%f222


/*     43 */	sxar2
/*     43 */	faddd,s	%f136,%f208,%f136
/*     43 */	faddd,s	%f140,%f210,%f140


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg16+-3920],%f224
/*     43 */	ldd,s	[%xg16+-3904],%f226


/*    161 */	sxar2
/*    161 */	faddd,s	%f144,%f212,%f144
/*    161 */	add	%l0,%xg13,%xg13


/*     43 */	sxar2
/*     43 */	add	%l0,%xg15,%xg15
/*     43 */	faddd,s	%f148,%f214,%f148


/*    161 */	sxar2
/*    161 */	add	%l0,%xg14,%xg14
/*    161 */	add	%l0,%xg16,%xg16


/*     43 */	sxar2
/*     43 */	faddd,s	%f152,%f216,%f152
/*     43 */	faddd,s	%f156,%f218,%f156


/*     43 */	sxar2
/*     43 */	faddd,s	%f160,%f220,%f160
/*     43 */	faddd,s	%f164,%f222,%f164


/*     43 */	sxar2
/*     43 */	faddd,s	%f168,%f224,%f168
/*     43 */	faddd,s	%f172,%f226,%f172

/*    161 */	bpos,pt	%icc, .L5911
	nop


.L5907:


/*    161 */	sxar2
/*    161 */	faddd,s	%f136,%f172,%f136
/*    161 */	faddd,s	%f64,%f100,%f64


/*    161 */	sxar2
/*    161 */	faddd,s	%f132,%f168,%f132
/*    161 */	faddd,s	%f60,%f96,%f60


/*    161 */	sxar2
/*    161 */	faddd,s	%f128,%f164,%f128
/*    161 */	faddd,s	%f56,%f92,%f56


/*    161 */	sxar2
/*    161 */	faddd,s	%f124,%f160,%f124
/*    161 */	faddd,s	%f52,%f88,%f52


/*    161 */	sxar2
/*    161 */	faddd,s	%f120,%f156,%f120
/*    161 */	faddd,s	%f48,%f84,%f48


/*    161 */	sxar2
/*    161 */	faddd,s	%f116,%f152,%f116
/*    161 */	faddd,s	%f44,%f80,%f44


/*    161 */	sxar2
/*    161 */	faddd,s	%f112,%f148,%f112
/*    161 */	faddd,s	%f40,%f76,%f40


/*    161 */	sxar2
/*    161 */	faddd,s	%f108,%f144,%f108
/*    161 */	faddd,s	%f36,%f72,%f36


/*    161 */	sxar2
/*    161 */	faddd,s	%f104,%f140,%f104
/*    161 */	faddd,s	%f32,%f68,%f32


/*    161 */	sxar2
/*    161 */	faddd,s	%f64,%f136,%f64
/*    161 */	faddd,s	%f60,%f132,%f60


/*    161 */	sxar2
/*    161 */	faddd,s	%f56,%f128,%f56
/*    161 */	faddd,s	%f52,%f124,%f52


/*    161 */	sxar2
/*    161 */	faddd,s	%f48,%f120,%f48
/*    161 */	faddd,s	%f44,%f116,%f44


/*    161 */	sxar2
/*    161 */	faddd,s	%f40,%f112,%f40
/*    161 */	faddd,s	%f36,%f108,%f36

/*    161 */	sxar1
/*    161 */	faddd,s	%f32,%f104,%f32

.L5774:

/*    150 */	sxar1
/*    150 */	addcc	%xg10,3,%xg10

/*    150 */	bneg	.L5767
	nop


.L5775:

/*    150 */	sxar1
/*    150 */	add	%l2,%xg12,%xg12

.L5782:


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg12],%f228
/*     43 */	ldd,s	[%xg12+16],%f230


/*     43 */	sxar2
/*     43 */	subcc	%xg10,1,%xg10
/*     43 */	ldd,s	[%xg12+32],%f232


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg12+48],%f234
/*     43 */	ldd,s	[%xg12+64],%f236


/*     43 */	sxar2
/*     43 */	faddd,s	%f32,%f228,%f32
/*     43 */	faddd,s	%f36,%f230,%f36


/*     43 */	sxar2
/*     43 */	faddd,s	%f40,%f232,%f40
/*     43 */	faddd,s	%f44,%f234,%f44


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg12+80],%f238
/*     43 */	ldd,s	[%xg12+96],%f240


/*     43 */	sxar2
/*     43 */	faddd,s	%f48,%f236,%f48
/*     43 */	ldd,s	[%xg12+112],%f242


/*    161 */	sxar2
/*    161 */	ldd,s	[%xg12+128],%f244
/*    161 */	add	%i4,%xg12,%xg12


/*     43 */	sxar2
/*     43 */	faddd,s	%f52,%f238,%f52
/*     43 */	faddd,s	%f56,%f240,%f56


/*     43 */	sxar2
/*     43 */	faddd,s	%f60,%f242,%f60
/*     43 */	faddd,s	%f64,%f244,%f64

/*    161 */	bpos,pt	%icc, .L5782
	nop


.L5776:


.L5767:


/*    161 */	sxar2
/*    161 */	std,s	%f32,[%fp+1839]
/*    161 */	std,s	%f36,[%fp+1855]


/*    161 */	sxar2
/*    161 */	std,s	%f40,[%fp+1871]
/*    161 */	std,s	%f44,[%fp+1887]


/*    161 */	sxar2
/*    161 */	std,s	%f48,[%fp+1903]
/*    161 */	std,s	%f52,[%fp+1919]


/*    161 */	sxar2
/*    161 */	std,s	%f56,[%fp+1935]
/*    161 */	std,s	%f60,[%fp+1951]

/*    161 */	sxar1
/*    161 */	std,s	%f64,[%fp+1967]

.L5768:



/*    181 */	sxar2
/*    181 */	ldd,s	[%fp+1839],%f246
/*    181 */	add	%xg7,2,%xg7



/*     81 */	sxar2
/*     81 */	subcc	%xg0,1,%xg0
/*     81 */	std	%f246,[%xg4]



/*     81 */	sxar2
/*     81 */	ldd,s	[%fp+1855],%f248
/*     81 */	std	%f248,[%xg4+8]



/*     81 */	sxar2
/*     81 */	ldd,s	[%fp+1871],%f250
/*     81 */	std	%f250,[%xg4+16]



/*     81 */	sxar2
/*     81 */	ldd,s	[%fp+1887],%f252
/*     81 */	std	%f252,[%xg4+24]



/*     81 */	sxar2
/*     81 */	ldd,s	[%fp+1903],%f254
/*     81 */	std	%f254,[%xg4+32]



/*     81 */	sxar2
/*     81 */	ldd,s	[%fp+1919],%f32
/*     81 */	std	%f32,[%xg4+40]



/*     81 */	sxar2
/*     81 */	ldd,s	[%fp+1935],%f34
/*     81 */	std	%f34,[%xg4+48]



/*     81 */	sxar2
/*     81 */	ldd,s	[%fp+1951],%f36
/*     81 */	std	%f36,[%xg4+56]



/*     81 */	sxar2
/*     81 */	ldd,s	[%fp+1967],%f38
/*     81 */	std	%f38,[%xg4+64]


/*     84 */	sxar2
/*     84 */	add	%xg4,144,%xg4
/*     84 */	std	%f502,[%xg8]


/*     84 */	sxar2
/*     84 */	std	%f504,[%xg8+8]
/*     84 */	std	%f506,[%xg8+16]


/*     84 */	sxar2
/*     84 */	std	%f508,[%xg8+24]
/*     84 */	std	%f510,[%xg8+32]


/*     84 */	sxar2
/*     84 */	std	%f288,[%xg8+40]
/*     84 */	std	%f290,[%xg8+48]


/*     84 */	sxar2
/*     84 */	std	%f292,[%xg8+56]
/*     84 */	std	%f294,[%xg8+64]

/*    181 */	sxar1
/*    181 */	add	%xg8,144,%xg8

/*    181 */	bne,pt	%icc, .L5763
/*    181 */	add	%o0,2,%o0


.L5769:

/*    181 */
/*    181 */	ba	.L5761
	nop


.L5771:

/*    181 *//*    181 */	call	__mpc_obar
/*    181 */	ldx	[%fp+2199],%o0

/*    181 *//*    181 */	call	__mpc_obar
/*    181 */	ldx	[%fp+2199],%o0


.L5772:

/*    181 */	ret
	restore



.LLFE8:
	.size	_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_3,.-_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_3
	.type	_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_3,#function
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite6-k.cpp _ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd $"
	.section	".text"
	.global	_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd
	.align	64
_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd:
.LLFB9:
.L630:

/*    184 */	save	%sp,-880,%sp
.LLCFI7:
/*    184 */	stw	%i0,[%fp+2179]
/*    184 */	std	%f2,[%fp+2183]
/*    184 */	stx	%i2,[%fp+2191]
/*    184 */	stx	%i3,[%fp+2199]
/*    184 */	stx	%i4,[%fp+2207]

.L631:

/*    192 *//*    192 */	sethi	%h44(_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_4),%o0
/*    192 */	mov	%fp,%l0
/*    192 */	or	%o0,%m44(_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_4),%o0
/*    192 */	mov	%g0,%l1
/*    192 */	sllx	%o0,12,%o0
/*    192 */	mov	%l0,%o1
/*    192 */	or	%o0,%l44(_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_4),%o0
/*    192 */	call	__mpc_opar
/*    192 */	mov	%l1,%o2

/*    202 */
/*    204 *//*    204 */	sethi	%h44(_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_5),%o0
/*    204 */	mov	%l0,%o1
/*    204 */	or	%o0,%m44(_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_5),%o0
/*    204 */	mov	%l1,%o2
/*    204 */	sllx	%o0,12,%o0
/*    204 */	call	__mpc_opar
/*    204 */	or	%o0,%l44(_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_5),%o0

/*    226 */
/*    226 */	ret
	restore



.L675:


.LLFE9:
	.size	_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd,.-_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd
	.type	_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd,#function
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite6-k.cpp _ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_4 $"
	.section	".text"
	.align	64
_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_4:
.LLFB10:
.L5784:

/*    192 */	save	%sp,-640,%sp
.LLCFI8:
/*    192 */	stx	%i0,[%fp+2175]
/*    192 */	stx	%i3,[%fp+2199]
/*    192 */	stx	%i0,[%fp+2175]

.L5785:

/*    192 *//*    192 */	sxar1
/*    192 */	ldsw	[%i0+2031],%xg9
/*    192 */
/*    192 */
/*    192 */
/*    193 */	ldsw	[%i0+2179],%l0
/*    193 */	cmp	%l0,%g0
/*    193 */	ble	.L5792
/*    193 */	mov	%g0,%o0


.L5786:

/*    193 */	sxar1
/*    193 */	mov	1,%xg8

/*    193 */	mov	1,%l5

/*    193 */	sxar1
/*    193 */	stx	%xg8,[%fp+2031]

/*    193 */	sra	%l0,%g0,%l0

/*    193 */	add	%fp,2039,%l1

/*    193 */	add	%fp,2023,%l2

/*    193 */	add	%fp,2031,%l3

/*    193 */	sra	%l5,%g0,%l4

.L5787:

/*    193 */	sra	%o0,%g0,%o0

/*    193 */	stx	%g0,[%sp+2223]

/*    193 */	mov	2,%o2

/*    193 */	mov	%g0,%o3

/*    193 */	mov	%l0,%o1

/*    193 */	mov	%l1,%o4


/*    193 */	stx	%g0,[%sp+2231]

/*    193 */	stx	%l3,[%sp+2239]


/*    193 */	sxar2
/*    193 */	ldx	[%fp+2199],%xg6
/*    193 */	stx	%xg6,[%sp+2247]

/*    193 */	call	__mpc_ostd_th
/*    193 */	mov	%l2,%o5
/*    193 */	sxar2
/*    193 */	ldx	[%fp+2031],%xg7
/*    193 */	cmp	%xg7,%g0
/*    193 */	ble,pn	%xcc, .L5792
	nop


.L5788:

/*    193 */	ldx	[%fp+2039],%o0

/*    193 */	sxar1
/*    193 */	ldx	[%fp+2023],%xg0

/*    193 */	sra	%o0,%g0,%o0


/*    193 */	sxar2
/*    193 */	sra	%xg0,%g0,%xg0
/*    193 */	sub	%xg0,%o0,%xg0


/*    193 */	sxar2
/*    193 */	add	%o0,1,%xg1
/*    193 */	srl	%xg0,31,%xg2


/*    193 */	sxar2
/*    193 */	sra	%o0,%g0,%xg3
/*    193 */	add	%xg0,%xg2,%xg0


/*    193 */	sxar2
/*    193 */	sra	%xg1,%g0,%xg1
/*    193 */	sra	%xg0,1,%xg0


/*    193 */	sxar2
/*    193 */	sllx	%xg3,5,%xg3
/*    193 */	add	%xg0,1,%xg0


/*    193 */	sxar2
/*    193 */	sllx	%xg1,5,%xg1
/*    193 */	sra	%xg0,%g0,%xg0


/*    193 */	sxar2
/*    193 */	sub	%l4,%xg0,%xg0
/*    193 */	srax	%xg0,32,%xg4


/*    193 */	sxar2
/*    193 */	and	%xg0,%xg4,%xg0
/*    193 */	sub	%l5,%xg0,%xg0

/*    193 */	sxar1
/*    193 */	subcc	%xg0,4,%xg0

/*    193 */	bneg	.L5795
	nop


.L5798:


/*    193 */	sxar2
/*    193 */	ldx	[%i0+2191],%xg6
/*    193 */	ldx	[%i0+2199],%xg10


/*    193 */	sxar2
/*    193 */	cmp	%xg0,16
/*    193 */	add	%xg6,16,%xg5


/*    193 */	sxar2
/*    193 */	add	%xg10,%xg3,%xg9
/*    193 */	add	%xg6,32,%xg7


/*    193 */	sxar2
/*    193 */	add	%xg6,48,%xg8
/*    193 */	add	%xg10,%xg1,%xg10

/*    193 */	bl	.L5918
	nop


.L5914:


.L5921:


/*    193 */	sxar2
/*    193 */	srl	%o0,31,%xg11
/*    193 */	add	%o0,2,%xg12


/*    193 */	sxar2
/*    193 */	add	%xg11,%o0,%xg11
/*    193 */	srl	%xg12,31,%xg13


/*    193 */	sxar2
/*    193 */	sra	%xg11,1,%xg11
/*    193 */	add	%xg12,%xg13,%xg13


/*    193 */	sxar2
/*    193 */	sra	%xg11,%g0,%xg11
/*    193 */	sra	%xg13,1,%xg13


/*    193 */	sxar2
/*    193 */	sllx	%xg11,2,%xg14
/*    193 */	add	%xg14,%xg11,%xg14


/*    193 */	sxar2
/*    193 */	sllx	%xg14,6,%xg14
/*    193 */	add	%xg14,%xg6,%xg15

.L5789:


/*    193 */	sxar2
/*    193 */	add	%xg14,%xg5,%o2
/*    193 */	sra	%xg13,%g0,%xg13


/*    193 */	sxar2
/*    193 */	ldd,s	[%xg15],%f80
/*    193 */	add	%xg14,%xg7,%o3


/*    193 */	sxar2
/*    193 */	ldd,s	[%o2],%f84
/*    193 */	sllx	%xg13,2,%o4


/*    193 */	sxar2
/*    193 */	add	%xg14,%xg8,%xg14
/*    193 */	add	%o4,%xg13,%o4


/*    193 */	sxar2
/*    193 */	ldd,s	[%o3],%f86
/*    193 */	ldd,s	[%xg14],%f90

/*    193 */	sllx	%o4,6,%o4


/*    193 */	sxar2
/*    193 */	add	%xg12,2,%o5
/*    193 */	add	%o4,%xg6,%o7


/*    193 */	sxar2
/*    193 */	srl	%o5,31,%xg2
/*    193 */	add	%o4,%xg5,%xg4



/*    193 */	sxar2
/*    193 */	add	%o5,%xg2,%o5
/*    193 */	fmovd	%f336,%f82



/*    193 */	sxar2
/*    193 */	fmovd	%f340,%f338
/*    193 */	fmovd	%f84,%f336


/*    193 */	sxar2
/*    193 */	ldd,s	[%o7],%f92
/*    193 */	add	%o4,%xg7,%xg11

/*    193 */	sxar1
/*    193 */	ldd,s	[%xg4],%f96

/*    193 */	sra	%o5,1,%o5


/*    193 */	sxar1
/*    193 */	add	%o4,%xg8,%o4

/*    193 */	sra	%o5,%g0,%o5




/*    193 */	sxar2
/*    193 */	fmovd	%f342,%f88
/*    193 */	fmovd	%f346,%f344


/*    193 */	sxar2
/*    193 */	fmovd	%f90,%f342
/*    193 */	std,s	%f80,[%xg9]


/*    193 */	sxar2
/*    193 */	sllx	%o5,2,%xg13
/*    193 */	add	%xg12,4,%xg14



/*    193 */	sxar2
/*    193 */	ldd,s	[%xg11],%f98
/*    193 */	ldd,s	[%o4],%f102


/*    193 */	sxar2
/*    193 */	add	%xg13,%o5,%xg13
/*    193 */	srl	%xg14,31,%xg15



/*    193 */	sxar2
/*    193 */	std,s	%f86,[%xg9+16]
/*    193 */	sllx	%xg13,6,%xg13




/*    193 */	sxar2
/*    193 */	add	%xg14,%xg15,%xg14
/*    193 */	fmovd	%f348,%f94



/*    193 */	sxar2
/*    193 */	fmovd	%f352,%f350
/*    193 */	fmovd	%f96,%f348


/*    193 */	sxar2
/*    193 */	std,s	%f82,[%xg10]
/*    193 */	add	%xg13,%xg6,%xg16



/*    193 */	sxar2
/*    193 */	sra	%xg14,1,%xg14
/*    193 */	std,s	%f88,[%xg10+16]


/*    193 */	sxar2
/*    193 */	add	%xg13,%xg5,%xg17
/*    193 */	sra	%xg14,%g0,%xg14


/*    193 */	sxar2
/*    193 */	std,s	%f92,[%xg9+64]
/*    193 */	add	%xg13,%xg7,%xg18




/*    193 */	sxar2
/*    193 */	sllx	%xg14,2,%xg19
/*    193 */	ldd,s	[%xg16],%f104


/*    193 */	sxar2
/*    193 */	add	%xg13,%xg8,%xg13
/*    193 */	ldd,s	[%xg17],%f108


/*    193 */	sxar2
/*    193 */	add	%xg19,%xg14,%xg19
/*    193 */	fmovd	%f98,%f100



/*    193 */	sxar2
/*    193 */	fmovd	%f102,%f356
/*    193 */	ldd,s	[%xg18],%f110


/*    193 */	sxar2
/*    193 */	add	%xg12,6,%xg20
/*    193 */	ldd,s	[%xg13],%f114



/*    193 */	sxar2
/*    193 */	sllx	%xg19,6,%xg19
/*    193 */	std,s	%f100,[%xg9+80]


/*    193 */	sxar2
/*    193 */	srl	%xg20,31,%xg21
/*    193 */	add	%xg19,%xg6,%xg22


/*    193 */	sxar2
/*    193 */	std,s	%f94,[%xg10+64]
/*    193 */	add	%xg19,%xg5,%xg23


/*    193 */	sxar2
/*    193 */	add	%xg19,%xg7,%xg24
/*    193 */	ldd,s	[%xg22],%f116


/*    193 */	sxar2
/*    193 */	add	%xg19,%xg8,%xg19
/*    193 */	ldd,s	[%xg23],%f120




/*    193 */	sxar2
/*    193 */	fmovd	%f354,%f102
/*    193 */	fmovd	%f360,%f106



/*    193 */	sxar2
/*    193 */	fmovd	%f364,%f362
/*    193 */	ldd,s	[%xg24],%f122


/*    193 */	sxar2
/*    193 */	ldd,s	[%xg19],%f126
/*    193 */	fmovd	%f108,%f360





/*    193 */	sxar2
/*    193 */	fmovd	%f366,%f112
/*    193 */	fmovd	%f370,%f368


/*    193 */	sxar2
/*    193 */	fmovd	%f114,%f366
/*    193 */	std,s	%f102,[%xg10+80]



/*    193 */	sxar2
/*    193 */	add	%xg21,%xg20,%xg21
/*    193 */	std,s	%f104,[%xg9+128]



/*    193 */	sxar2
/*    193 */	sra	%xg21,1,%xg21
/*    193 */	std,s	%f110,[%xg9+144]


/*    193 */	sxar2
/*    193 */	sra	%xg21,%g0,%xg21
/*    193 */	add	%xg12,8,%xg25





/*    193 */	sxar2
/*    193 */	std,s	%f106,[%xg10+128]
/*    193 */	sllx	%xg21,2,%xg26


/*    193 */	sxar2
/*    193 */	srl	%xg25,31,%xg27
/*    193 */	fmovd	%f116,%f118





/*    193 */	sxar2
/*    193 */	fmovd	%f120,%f374
/*    193 */	std,s	%f112,[%xg10+144]


/*    193 */	sxar2
/*    193 */	add	%xg26,%xg21,%xg26
/*    193 */	add	%xg25,%xg27,%xg25


/*    193 */	sxar2
/*    193 */	fmovd	%f122,%f124
/*    193 */	fmovd	%f126,%f380



/*    193 */	sxar2
/*    193 */	std,s	%f118,[%xg9+192]
/*    193 */	sllx	%xg26,6,%xg26


/*    193 */	sxar2
/*    193 */	sra	%xg25,1,%xg25
/*    193 */	fmovd	%f372,%f120



/*    193 */	sxar2
/*    193 */	std,s	%f124,[%xg9+208]
/*    193 */	add	%xg26,%xg6,%xg28



/*    193 */	sxar2
/*    193 */	fmovd	%f378,%f126
/*    193 */	std,s	%f120,[%xg10+192]


/*    193 */	sxar2
/*    193 */	std,s	%f126,[%xg10+208]
/*    193 */	add	%xg26,%xg5,%xg29


/*    193 */	sxar2
/*    193 */	sra	%xg25,%g0,%xg25
/*    193 */	ldd,s	[%xg28],%f128


/*    193 */	sxar2
/*    193 */	add	%xg26,%xg7,%xg30
/*    193 */	ldd,s	[%xg29],%f132


/*    193 */	sxar2
/*    193 */	sllx	%xg25,2,%xg31
/*    193 */	add	%xg26,%xg8,%xg26


/*    193 */	sxar2
/*    193 */	add	%xg31,%xg25,%xg31
/*    193 */	ldd,s	[%xg30],%f134


/*    193 */	sxar2
/*    193 */	ldd,s	[%xg26],%f138
/*    193 */	sllx	%xg31,6,%xg31


/*    193 */	sxar2
/*    193 */	add	%xg12,10,%g1
/*    193 */	add	%xg31,%xg6,%g2

/*    193 */	srl	%g1,31,%g3

/*    193 */	sxar1
/*    193 */	add	%xg31,%xg5,%g4

/*    193 */	add	%g1,%g3,%g1




/*    193 */	sxar2
/*    193 */	ldd,s	[%g2],%f140
/*    193 */	add	%xg31,%xg7,%g5

/*    193 */	sxar1
/*    193 */	ldd,s	[%g4],%f144

/*    193 */	sra	%g1,1,%g1


/*    193 */	sxar2
/*    193 */	fmovd	%f128,%f130
/*    193 */	fmovd	%f132,%f386


/*    193 */	sxar1
/*    193 */	add	%xg31,%xg8,%xg31

/*    193 */	sra	%g1,%g0,%g1



/*    193 */	sxar1
/*    193 */	std,s	%f130,[%xg9+256]

/*    193 */	sllx	%g1,2,%o0


/*    193 */	sxar2
/*    193 */	add	%xg12,12,%o1
/*    193 */	fmovd	%f134,%f136



/*    193 */	sxar2
/*    193 */	fmovd	%f138,%f392
/*    193 */	ldd,s	[%g5],%f146

/*    193 */	sxar1
/*    193 */	ldd,s	[%xg31],%f150

/*    193 */	add	%o0,%g1,%o0

/*    193 */	srl	%o1,31,%o2



/*    193 */	sxar2
/*    193 */	fmovd	%f384,%f132
/*    193 */	std,s	%f136,[%xg9+272]

/*    193 */	sllx	%o0,6,%o0

/*    193 */	add	%o1,%o2,%o1




/*    193 */	sxar2
/*    193 */	fmovd	%f390,%f138
/*    193 */	fmovd	%f396,%f142



/*    193 */	sxar2
/*    193 */	fmovd	%f400,%f398
/*    193 */	fmovd	%f144,%f396


/*    193 */	sxar2
/*    193 */	std,s	%f132,[%xg10+256]
/*    193 */	add	%o0,%xg6,%o3

/*    193 */	sra	%o1,1,%o1



/*    193 */	sxar2
/*    193 */	std,s	%f138,[%xg10+272]
/*    193 */	add	%o0,%xg5,%o4

/*    193 */	sra	%o1,%g0,%o1


/*    193 */	sxar2
/*    193 */	std,s	%f140,[%xg9+320]
/*    193 */	add	%o0,%xg7,%o5

/*    193 */	sllx	%o1,2,%o7




/*    193 */	sxar2
/*    193 */	ldd,s	[%o3],%f152
/*    193 */	add	%o0,%xg8,%o0

/*    193 */	sxar1
/*    193 */	ldd,s	[%o4],%f156

/*    193 */	add	%o7,%o1,%o7


/*    193 */	sxar2
/*    193 */	fmovd	%f146,%f148
/*    193 */	fmovd	%f150,%f404



/*    193 */	sxar2
/*    193 */	ldd,s	[%o5],%f158
/*    193 */	add	%xg12,14,%xg2

/*    193 */	sxar1
/*    193 */	ldd,s	[%o0],%f162

/*    193 */	sllx	%o7,6,%o7



/*    193 */	sxar2
/*    193 */	std,s	%f148,[%xg9+336]
/*    193 */	srl	%xg2,31,%xg4


/*    193 */	sxar2
/*    193 */	add	%o7,%xg6,%xg11
/*    193 */	std,s	%f142,[%xg10+320]


/*    193 */	sxar2
/*    193 */	add	%o7,%xg5,%xg13
/*    193 */	add	%o7,%xg7,%xg14


/*    193 */	sxar2
/*    193 */	ldd,s	[%xg11],%f164
/*    193 */	add	%o7,%xg8,%o7


/*    193 */	sxar2
/*    193 */	ldd,s	[%xg13],%f168
/*    193 */	fmovd	%f402,%f150





/*    193 */	sxar2
/*    193 */	ldd,s	[%xg14],%f170
/*    193 */	ldd,s	[%o7],%f174


/*    193 */	sxar2
/*    193 */	fmovd	%f152,%f154
/*    193 */	fmovd	%f156,%f410





/*    193 */	sxar2
/*    193 */	std,s	%f150,[%xg10+336]
/*    193 */	add	%xg4,%xg2,%xg4


/*    193 */	sxar2
/*    193 */	fmovd	%f158,%f160
/*    193 */	fmovd	%f162,%f416



/*    193 */	sxar2
/*    193 */	std,s	%f154,[%xg9+384]
/*    193 */	sra	%xg4,1,%xg4



/*    193 */	sxar2
/*    193 */	fmovd	%f408,%f156
/*    193 */	std,s	%f160,[%xg9+400]


/*    193 */	sxar2
/*    193 */	sra	%xg4,%g0,%xg4
/*    193 */	add	%xg12,16,%xg12





/*    193 */	sxar2
/*    193 */	fmovd	%f414,%f162
/*    193 */	std,s	%f156,[%xg10+384]


/*    193 */	sxar2
/*    193 */	sllx	%xg4,2,%xg14
/*    193 */	srl	%xg12,31,%xg13


/*    193 */	sxar2
/*    193 */	fmovd	%f164,%f166
/*    193 */	fmovd	%f168,%f422





/*    193 */	sxar2
/*    193 */	fmovd	%f426,%f172
/*    193 */	fmovd	%f430,%f428


/*    193 */	sxar2
/*    193 */	fmovd	%f174,%f426
/*    193 */	std,s	%f162,[%xg10+400]


/*    193 */	sxar2
/*    193 */	add	%xg14,%xg4,%xg14
/*    193 */	add	%xg12,%xg13,%xg13



/*    193 */	sxar2
/*    193 */	std,s	%f166,[%xg9+448]
/*    193 */	sllx	%xg14,6,%xg14


/*    193 */	sxar2
/*    193 */	sra	%xg13,1,%xg13
/*    193 */	fmovd	%f420,%f168



/*    193 */	sxar2
/*    193 */	std,s	%f170,[%xg9+464]
/*    193 */	add	%xg14,%xg6,%xg15



/*    193 */	sxar2
/*    193 */	add	%xg3,512,%xg3
/*    193 */	std,s	%f168,[%xg10+448]


/*    193 */	sxar2
/*    193 */	add	%xg9,512,%xg9
/*    193 */	add	%xg1,512,%xg1


/*    193 */	sxar2
/*    193 */	std,s	%f172,[%xg10+464]
/*    193 */	add	%xg10,512,%xg10


/*    193 */	sxar2
/*    193 */	sub	%xg0,8,%xg0
/*    193 */	cmp	%xg0,19

/*    193 */	bge,pt	%icc, .L5789
	nop


.L5922:


/*    193 */	sxar2
/*    193 */	add	%xg14,%xg5,%xg16
/*    193 */	ldd,s	[%xg15],%f32


/*    193 */	sxar2
/*    193 */	add	%xg14,%xg7,%xg17
/*    193 */	sra	%xg13,%g0,%xg13


/*    193 */	sxar2
/*    193 */	ldd,s	[%xg16],%f36
/*    193 */	add	%xg14,%xg8,%xg14


/*    193 */	sxar2
/*    193 */	ldd,s	[%xg17],%f38
/*    193 */	sllx	%xg13,2,%xg18


/*    193 */	sxar2
/*    193 */	ldd,s	[%xg14],%f42
/*    193 */	add	%xg12,2,%xg19


/*    193 */	sxar2
/*    193 */	add	%xg18,%xg13,%xg18
/*    193 */	srl	%xg19,31,%xg20


/*    193 */	sxar2
/*    193 */	sllx	%xg18,6,%xg18
/*    193 */	add	%xg19,%xg20,%xg19


/*    193 */	fmovd	%f32,%f34


/*    193 */	sxar2
/*    193 */	add	%xg18,%xg6,%xg21
/*    193 */	add	%xg18,%xg5,%xg22





/*    193 */	sxar2
/*    193 */	fmovd	%f36,%f290
/*    193 */	fmovd	%f294,%f40


/*    193 */	sxar2
/*    193 */	add	%xg18,%xg7,%xg23
/*    193 */	add	%xg18,%xg8,%xg18



/*    193 */	sxar2
/*    193 */	fmovd	%f298,%f296
/*    193 */	fmovd	%f42,%f294




/*    193 */	sxar2
/*    193 */	ldd,s	[%xg21],%f44
/*    193 */	fmovd	%f288,%f36



/*    193 */	sxar2
/*    193 */	sra	%xg19,1,%xg19
/*    193 */	add	%xg12,4,%xg24


/*    193 */	sxar2
/*    193 */	sra	%xg19,%g0,%xg19
/*    193 */	srl	%xg24,31,%xg25


/*    193 */	sxar2
/*    193 */	ldd,s	[%xg22],%f48
/*    193 */	ldd,s	[%xg23],%f50


/*    193 */	sxar2
/*    193 */	sllx	%xg19,2,%xg26
/*    193 */	add	%xg24,%xg25,%xg24


/*    193 */	sxar2
/*    193 */	ldd,s	[%xg18],%f54
/*    193 */	add	%xg26,%xg19,%xg26


/*    193 */	sxar2
/*    193 */	sra	%xg24,1,%xg24
/*    193 */	std,s	%f34,[%xg9]


/*    193 */	fmovd	%f44,%f46


/*    193 */	sxar2
/*    193 */	sllx	%xg26,6,%xg26
/*    193 */	sra	%xg24,%g0,%xg24


/*    193 */	sxar2
/*    193 */	std,s	%f38,[%xg9+16]
/*    193 */	add	%xg26,%xg6,%xg27


/*    193 */	sxar1
/*    193 */	fmovd	%f48,%f302


/*    193 */	fmovd	%f50,%f52



/*    193 */	sxar2
/*    193 */	add	%xg26,%xg5,%xg28
/*    193 */	add	%xg26,%xg7,%xg29




/*    193 */	sxar2
/*    193 */	fmovd	%f300,%f48
/*    193 */	fmovd	%f54,%f308



/*    193 */	sxar2
/*    193 */	std,s	%f36,[%xg10]
/*    193 */	add	%xg26,%xg8,%xg26


/*    193 */	sxar2
/*    193 */	sllx	%xg24,2,%xg30
/*    193 */	fmovd	%f306,%f54



/*    193 */	sxar2
/*    193 */	std,s	%f40,[%xg10+16]
/*    193 */	add	%xg12,6,%o0


/*    193 */	sxar2
/*    193 */	add	%xg30,%xg24,%xg30
/*    193 */	ldd,s	[%xg27],%f60


/*    193 */	sxar2
/*    193 */	ldd,s	[%xg28],%f64
/*    193 */	add	%xg3,256,%xg3


/*    193 */	sxar2
/*    193 */	sllx	%xg30,6,%xg30
/*    193 */	ldd,s	[%xg29],%f56


/*    193 */	sxar2
/*    193 */	ldd,s	[%xg26],%f66
/*    193 */	add	%xg30,%xg6,%xg31


/*    193 */	sxar2
/*    193 */	add	%xg30,%xg5,%g1
/*    193 */	std,s	%f46,[%xg9+64]


/*    193 */	sxar2
/*    193 */	add	%xg30,%xg7,%g2
/*    193 */	add	%xg30,%xg8,%xg30


/*    193 */	sxar2
/*    193 */	std,s	%f52,[%xg9+80]
/*    193 */	add	%xg1,256,%xg1

/*    193 */	sxar1
/*    193 */	sub	%xg0,4,%xg0



/*    193 */	fmovd	%f60,%f62



/*    193 */	sxar2
/*    193 */	fmovd	%f64,%f318
/*    193 */	ldd,s	[%xg31],%f68


/*    193 */	fmovd	%f56,%f58



/*    193 */	sxar2
/*    193 */	ldd,s	[%g1],%f72
/*    193 */	fmovd	%f66,%f314



/*    193 */	sxar2
/*    193 */	std,s	%f48,[%xg10+64]
/*    193 */	fmovd	%f316,%f64




/*    193 */	sxar2
/*    193 */	fmovd	%f312,%f66
/*    193 */	std,s	%f54,[%xg10+80]


/*    193 */	sxar2
/*    193 */	ldd,s	[%g2],%f74
/*    193 */	ldd,s	[%xg30],%f78




/*    193 */	sxar2
/*    193 */	fmovd	%f68,%f70
/*    193 */	fmovd	%f72,%f326




/*    193 */	sxar2
/*    193 */	fmovd	%f324,%f72
/*    193 */	std,s	%f62,[%xg9+128]



/*    193 */	sxar2
/*    193 */	std,s	%f58,[%xg9+144]
/*    193 */	fmovd	%f74,%f76




/*    193 */	sxar2
/*    193 */	fmovd	%f78,%f332
/*    193 */	std,s	%f64,[%xg10+128]



/*    193 */	sxar2
/*    193 */	fmovd	%f330,%f78
/*    193 */	std,s	%f66,[%xg10+144]


/*    193 */	sxar2
/*    193 */	std,s	%f70,[%xg9+192]
/*    193 */	std,s	%f76,[%xg9+208]


/*    193 */	sxar2
/*    193 */	add	%xg9,256,%xg9
/*    193 */	std,s	%f72,[%xg10+192]


/*    193 */	sxar2
/*    193 */	std,s	%f78,[%xg10+208]
/*    193 */	add	%xg10,256,%xg10

.L5918:


.L5917:


.L5920:


/*    202 */	sxar2
/*    202 */	srl	%o0,31,%xg15
/*    202 */	add	%o0,2,%xg16


/*    194 */	sxar2
/*    194 */	add	%xg15,%o0,%xg15
/*    194 */	srl	%xg16,31,%xg17


/*    194 */	sxar2
/*    194 */	sra	%xg15,1,%xg15
/*    194 */	add	%xg16,%xg17,%xg17


/*    194 */	sxar2
/*    194 */	sra	%xg15,%g0,%xg15
/*    194 */	sra	%xg17,1,%xg17


/*    194 */	sxar2
/*    194 */	sllx	%xg15,2,%xg18
/*    194 */	sra	%xg17,%g0,%xg17


/*    194 */	sxar2
/*    194 */	add	%xg18,%xg15,%xg18
/*    194 */	sllx	%xg17,2,%xg19


/*    194 */	sxar2
/*    194 */	sllx	%xg18,6,%xg18
/*    194 */	add	%xg19,%xg17,%xg19


/*    196 */	sxar2
/*    196 */	add	%xg18,%xg6,%xg20
/*    196 */	add	%xg18,%xg7,%xg22


/*    197 */	sxar2
/*    197 */	add	%xg18,%xg5,%xg21
/*    197 */	add	%xg18,%xg8,%xg18


/*    102 */	sxar2
/*    102 */	ldd,s	[%xg20],%f176
/*    102 */	ldd,s	[%xg22],%f182


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg21],%f180
/*    194 */	sllx	%xg19,6,%xg19


/*    202 */	sxar2
/*    202 */	ldd,s	[%xg18],%f186
/*    202 */	add	%xg16,2,%xg16


/*    196 */	sxar2
/*    196 */	add	%xg19,%xg6,%xg23
/*    196 */	add	%xg19,%xg7,%xg25


/*    102 */	sxar2
/*    102 */	add	%xg19,%xg5,%xg24
/*    102 */	ldd,s	[%xg23],%f188


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg25],%f194
/*    194 */	srl	%xg16,31,%xg26



/*    102 */	sxar2
/*    102 */	add	%xg19,%xg8,%xg19
/*    102 */	fmovd	%f432,%f178



/*    202 */	sxar2
/*    202 */	fmovd	%f438,%f184
/*    202 */	add	%xg16,2,%o0



/*    102 */	sxar2
/*    102 */	ldd,s	[%xg24],%f192
/*    102 */	fmovd	%f180,%f432





/*    194 */	sxar2
/*    194 */	fmovd	%f186,%f438
/*    194 */	add	%xg16,%xg26,%xg16


/*    102 */	sxar2
/*    102 */	fmovd	%f436,%f434
/*    102 */	ldd,s	[%xg19],%f198




/*    194 */	sxar2
/*    194 */	fmovd	%f442,%f440
/*    194 */	sra	%xg16,1,%xg16




/*    102 */	sxar2
/*    102 */	fmovd	%f444,%f190
/*    102 */	fmovd	%f450,%f196


/*    194 */	sxar2
/*    194 */	srl	%o0,31,%xg27
/*    194 */	sra	%xg16,%g0,%xg16


/*    194 */	sxar2
/*    194 */	add	%o0,%xg27,%xg27
/*    194 */	sllx	%xg16,2,%xg28



/*    102 */	sxar2
/*    102 */	fmovd	%f448,%f446
/*    102 */	fmovd	%f192,%f444




/*    194 */	sxar2
/*    194 */	sra	%xg27,1,%xg27
/*    194 */	add	%xg28,%xg16,%xg28



/*    102 */	sxar2
/*    102 */	fmovd	%f454,%f452
/*    102 */	fmovd	%f198,%f450




/*    194 */	sxar2
/*    194 */	sra	%xg27,%g0,%xg27
/*    194 */	sllx	%xg28,6,%xg28


/*    194 */	sxar2
/*    194 */	std,s	%f176,[%xg9]
/*    194 */	add	%xg28,%xg6,%xg30


/*     25 */	sxar2
/*     25 */	add	%xg28,%xg7,%g1
/*     25 */	std,s	%f182,[%xg9+16]


/*    197 */	sxar2
/*    197 */	add	%xg28,%xg5,%xg31
/*    197 */	add	%xg28,%xg8,%xg28


/*    194 */	sxar2
/*    194 */	std,s	%f178,[%xg10]
/*    194 */	sllx	%xg27,2,%xg29



/*    194 */	sxar2
/*    194 */	std,s	%f184,[%xg10+16]
/*    194 */	add	%xg29,%xg27,%xg29


/*    102 */	sxar2
/*    102 */	add	%xg3,256,%xg3
/*    102 */	ldd,s	[%xg30],%f200


/*    102 */	sxar2
/*    102 */	sllx	%xg29,6,%xg29
/*    102 */	ldd,s	[%g1],%f206


/*     24 */	sxar2
/*     24 */	add	%xg1,256,%xg1
/*     24 */	std,s	%f188,[%xg9+64]


/*    196 */	sxar2
/*    196 */	add	%xg29,%xg6,%g2
/*    196 */	add	%xg29,%xg7,%g4


/*    195 */	sxar2
/*    195 */	std,s	%f194,[%xg9+80]
/*    195 */	add	%xg29,%xg5,%g3


/*    102 */	sxar2
/*    102 */	add	%xg29,%xg8,%xg29
/*    102 */	ldd,s	[%xg31],%f204


/*    202 */	sxar2
/*    202 */	ldd,s	[%xg28],%f210
/*    202 */	subcc	%xg0,4,%xg0




/*    102 */	sxar2
/*    102 */	fmovd	%f456,%f202
/*    102 */	fmovd	%f462,%f208


/*     24 */	sxar2
/*     24 */	ldd,s	[%g2],%f212
/*     24 */	std,s	%f190,[%xg10+64]



/*    102 */	sxar2
/*    102 */	std,s	%f196,[%xg10+80]
/*    102 */	fmovd	%f204,%f456





/*    102 */	sxar2
/*    102 */	fmovd	%f210,%f462
/*    102 */	ldd,s	[%g4],%f218


/*    102 */	sxar2
/*    102 */	fmovd	%f460,%f458
/*    102 */	fmovd	%f466,%f464





/*    102 */	sxar2
/*    102 */	ldd,s	[%g3],%f216
/*    102 */	ldd,s	[%xg29],%f222



/*    102 */	sxar2
/*    102 */	fmovd	%f468,%f214
/*    102 */	fmovd	%f474,%f220



/*    102 */	sxar2
/*    102 */	fmovd	%f472,%f470
/*    102 */	fmovd	%f216,%f468





/*    102 */	sxar2
/*    102 */	std,s	%f200,[%xg9+128]
/*    102 */	fmovd	%f478,%f476




/*     25 */	sxar2
/*     25 */	fmovd	%f222,%f474
/*     25 */	std,s	%f206,[%xg9+144]


/*     25 */	sxar2
/*     25 */	std,s	%f202,[%xg10+128]
/*     25 */	std,s	%f208,[%xg10+144]


/*     25 */	sxar2
/*     25 */	std,s	%f212,[%xg9+192]
/*     25 */	std,s	%f218,[%xg9+208]


/*     24 */	sxar2
/*     24 */	add	%xg9,256,%xg9
/*     24 */	std,s	%f214,[%xg10+192]


/*    202 */	sxar2
/*    202 */	std,s	%f220,[%xg10+208]
/*    202 */	add	%xg10,256,%xg10

/*    202 */	bpos,pt	%icc, .L5920
/*    202 */	add	%o0,2,%o0


.L5916:


.L5795:

/*    193 */	sxar1
/*    193 */	addcc	%xg0,3,%xg0

/*    193 */	bneg	.L5790
	nop


.L5796:

/*    193 */	ldx	[%i0+2191],%g3

/*    193 */	ldx	[%i0+2199],%o4

/*    193 */	add	%g3,16,%g4

/*    193 */	add	%g3,32,%g5

/*    193 */	add	%g3,48,%o1

.L5803:

/*    194 */	srl	%o0,31,%o2

/*    200 */	sxar1
/*    200 */	add	%o4,%xg3,%o3

/*    194 */	add	%o2,%o0,%o2

/*    201 */	sxar1
/*    201 */	add	%o4,%xg1,%o5

/*    194 */	sra	%o2,1,%o2


/*    194 */	sra	%o2,%g0,%o2

/*    202 */	sxar1
/*    202 */	add	%xg1,64,%xg1

/*    194 */	sllx	%o2,2,%o7

/*    202 */	sxar1
/*    202 */	add	%xg3,64,%xg3

/*    194 */	add	%o7,%o2,%o7

/*    202 */	sxar1
/*    202 */	subcc	%xg0,1,%xg0

/*    194 */	sllx	%o7,6,%o7


/*    195 */	sxar2
/*    195 */	add	%o7,%g3,%xg2
/*    195 */	add	%o7,%g4,%xg4

/*    196 */	sxar1
/*    196 */	add	%o7,%g5,%xg5

/*    197 */	add	%o7,%o1,%o7


/*    102 */	sxar2
/*    102 */	ldd,s	[%xg2],%f224
/*    102 */	ldd,s	[%xg4],%f228


/*    102 */	sxar2
/*    102 */	ldd,s	[%o7],%f234
/*    102 */	ldd,s	[%xg5],%f230




/*    102 */	sxar2
/*    102 */	fmovd	%f224,%f226
/*    102 */	fmovd	%f228,%f482




/*    105 */	sxar2
/*    105 */	fmovd	%f234,%f488
/*    105 */	fmovd	%f480,%f228





/*    105 */	sxar2
/*    105 */	fmovd	%f230,%f232
/*    105 */	fmovd	%f486,%f234



/*     25 */	sxar2
/*     25 */	std,s	%f226,[%o3]
/*     25 */	std,s	%f232,[%o3+16]


/*     25 */	sxar2
/*     25 */	std,s	%f228,[%o5]
/*     25 */	std,s	%f234,[%o5+16]

/*    202 */	bpos,pt	%icc, .L5803
/*    202 */	add	%o0,2,%o0


.L5797:


.L5790:

/*    202 */
/*    202 */	ba	.L5787
	nop


.L5792:

/*    202 *//*    202 */	call	__mpc_obar
/*    202 */	ldx	[%fp+2199],%o0

/*    202 *//*    202 */	call	__mpc_obar
/*    202 */	ldx	[%fp+2199],%o0


.L5793:

/*    202 */	ret
	restore



.LLFE10:
	.size	_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_4,.-_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_4
	.type	_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_4,#function
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite6-k.cpp _ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_5 $"
	.section	".text"
	.align	64
_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_5:
.LLFB11:
.L5805:

/*    204 */	save	%sp,-2048,%sp
.LLCFI9:
/*    204 */	stx	%i0,[%fp+2175]
/*    204 */	stx	%i3,[%fp+2199]
/*    204 */	stx	%i0,[%fp+2175]

.L5806:

/*    204 *//*    204 */	ldsw	[%i0+2035],%g1
/*    204 */
/*    204 */
/*    204 */
/*    205 */	ldsw	[%i0+2179],%l0
/*    205 */	cmp	%l0,%g0
/*    205 */	ble	.L5820
/*    205 */	mov	%g0,%o0


.L5807:

/*    205 */	sxar1
/*    205 */	fzero,s	%f34

/*    205 */	sethi	%h44(.LR0.cnt.4),%g1

/*    205 */	sxar1
/*    205 */	sethi	%h44(.LR0.cnt.5),%xg0

/*    205 */	or	%g1,%m44(.LR0.cnt.4),%g1

/*    205 */	sxar1
/*    205 */	or	%xg0,%m44(.LR0.cnt.5),%xg0

/*    205 */	sllx	%g1,12,%g1

/*    205 */	sxar1
/*    205 */	sllx	%xg0,12,%xg0

/*    205 */	or	%g1,%l44(.LR0.cnt.4),%g1


/*    205 */	sxar2
/*    205 */	or	%xg0,%l44(.LR0.cnt.5),%xg0
/*    205 */	mov	1,%xg31

/*    205 */	sra	%l0,%g0,%l0


/*    205 */	sxar2
/*    205 */	ldd	[%g1],%f232
/*    205 */	ldd	[%g1],%f488



/*    205 */	sxar2
/*    205 */	ldd	[%xg0],%f234
/*    205 */	ldd	[%xg0],%f490



/*    205 */	sxar2
/*    ??? */	std,s	%f34,[%fp+223]
/*    205 */	stx	%xg31,[%fp+2031]


/*    205 */	sxar2
/*    ??? */	std,s	%f232,[%fp+255]
/*    ??? */	std,s	%f234,[%fp+239]

.L5837:

/*    205 */	add	%fp,2039,%l1

/*    205 */	mov	1,%l5

/*    205 */	add	%fp,2023,%l2

/*    205 */	add	%fp,2031,%l3

/*    205 */	sra	%l5,%g0,%l4

.L5809:

/*    205 */	sra	%o0,%g0,%o0

/*    205 */	stx	%g0,[%sp+2223]

/*    205 */	mov	4,%o2

/*    205 */	mov	%g0,%o3

/*    205 */	mov	%l0,%o1

/*    205 */	mov	%l1,%o4


/*    205 */	stx	%g0,[%sp+2231]

/*    205 */	stx	%l3,[%sp+2239]


/*    205 */	sxar2
/*    205 */	ldx	[%fp+2199],%xg29
/*    205 */	stx	%xg29,[%sp+2247]

/*    205 */	call	__mpc_ostd_th
/*    205 */	mov	%l2,%o5
/*    205 */	sxar2
/*    205 */	ldx	[%fp+2031],%xg30
/*    205 */	cmp	%xg30,%g0
/*    205 */	ble,pn	%xcc, .L5820
	nop


.L5810:

/*    205 */	ldx	[%fp+2039],%o0


/*    205 */	sxar2
/*    205 */	ldx	[%fp+2023],%xg0
/*    205 */	ldd	[%i0+2183],%f74



/*    205 */	sxar2
/*    205 */	ldd	[%i0+2183],%f330
/*    205 */	ldsw	[%i0+2179],%xg9


/*    205 */	sxar2
/*    205 */	ldx	[%i0+2199],%xg5
/*    205 */	ldx	[%i0+2207],%xg19

/*    205 */	sra	%o0,%g0,%o0


/*    205 */	sxar2
/*    205 */	sra	%xg0,%g0,%xg0
/*    205 */	sub	%xg0,%o0,%xg0


/*    205 */	sxar2
/*    205 */	sra	%o0,%g0,%xg1
/*    205 */	sra	%xg0,1,%xg2


/*    205 */	sxar2
/*    205 */	sllx	%xg1,5,%xg3
/*    205 */	srl	%xg2,30,%xg2


/*    205 */	sxar2
/*    205 */	sllx	%xg1,3,%xg1
/*    205 */	add	%xg0,%xg2,%xg0


/*    205 */	sxar2
/*    205 */	add	%xg5,32,%xg4
/*    205 */	sra	%xg0,2,%xg0


/*    205 */	sxar2
/*    205 */	add	%xg0,1,%xg0
/*    205 */	sra	%xg0,%g0,%xg0


/*    205 */	sxar2
/*    205 */	sub	%l4,%xg0,%xg0
/*    205 */	srax	%xg0,32,%xg6


/*    205 */	sxar2
/*    205 */	and	%xg0,%xg6,%xg0
/*    205 */	sub	%l5,%xg0,%xg0

.L5811:


/*    212 */	sxar2
/*    212 */	add	%xg5,%xg3,%xg7
/*    212 */	cmp	%xg9,%g0


/*    207 */	sxar2
/*    ??? */	ldd,s	[%fp+223],%f32
/*    207 */	ldd	[%xg7],%f250



/*    207 */	sxar2
/*    207 */	ldd	[%xg7+32],%f506
/*    207 */	ldd	[%xg7+64],%f38



/*     37 */	sxar2
/*     37 */	ldd	[%xg7+96],%f294
/*     37 */	std,s	%f250,[%fp+1135]


/*    208 */	sxar2
/*    208 */	std,s	%f38,[%fp+1151]
/*    208 */	ldd	[%xg7+8],%f252



/*    208 */	sxar2
/*    208 */	ldd	[%xg7+40],%f508
/*    208 */	ldd	[%xg7+72],%f46



/*     37 */	sxar2
/*     37 */	ldd	[%xg7+104],%f302
/*     37 */	std,s	%f252,[%fp+1167]


/*    209 */	sxar2
/*    209 */	std,s	%f46,[%fp+1183]
/*    209 */	ldd	[%xg7+16],%f254



/*    209 */	sxar2
/*    209 */	ldd	[%xg7+48],%f510
/*    209 */	ldd	[%xg7+80],%f56



/*     34 */	sxar2
/*     34 */	ldd	[%xg7+112],%f312
/*     34 */	std,s	%f74,[%fp+1231]


/*     34 */	sxar2
/*     34 */	std,s	%f74,[%fp+1247]
/*     34 */	std,s	%f32,[%fp+1263]


/*     37 */	sxar2
/*     37 */	std,s	%f254,[%fp+1199]
/*     37 */	std,s	%f56,[%fp+1215]

/*     34 */	sxar1
/*     34 */	std,s	%f32,[%fp+1279]

/*    212 */	ble	.L5817
	nop


.L5813:


/*    224 */	sxar2
/*    224 */	ldd,s	[%fp+1135],%f32
/*    224 */	mov	%g0,%xg10


/*    224 */	sxar2
/*    ??? */	ldd,s	[%fp+223],%f186
/*    224 */	subcc	%xg9,2,%xg8


/*    212 */	sxar2
/*    212 */	ldd,s	[%fp+1263],%f184
/*    212 */	ldd,s	[%fp+1167],%f42


/*    212 */	sxar2
/*    212 */	ldd,s	[%fp+1199],%f50
/*    212 */	ldd,s	[%fp+1231],%f72

/*    224 */	bneg	.L5823
	nop


.L5826:


/*    212 */	sxar2
/*    212 */	ldx	[%i0+2199],%xg12
/*    212 */	cmp	%xg8,14

/*    212 */	bl	.L5927
	nop


.L5923:


.L5930:


/*    212 */	sxar2
/*    212 */	add	%xg12,%xg10,%xg11
/*    212 */	add	%xg4,%xg10,%xg13


/*    212 */	sxar2
/*    ??? */	ldd,s	[%fp+255],%f158
/*    212 */	ldd,s	[%xg11],%f36


/*    212 */	sxar2
/*    212 */	ldd,s	[%xg11+16],%f52
/*    212 */	add	%xg10,64,%xg14


/*    212 */	sxar2
/*    212 */	add	%xg10,128,%xg10
/*    212 */	add	%xg12,%xg14,%xg15


/*    212 */	sxar2
/*    212 */	ldd,s	[%xg13],%f60
/*    212 */	ldd,s	[%xg13+16],%f68


/*    212 */	sxar2
/*    212 */	add	%xg4,%xg14,%xg14
/*    212 */	ldd,s	[%xg15],%f78


/*    212 */	sxar2
/*    212 */	add	%xg12,%xg10,%xg16
/*    ??? */	ldd,s	[%fp+239],%f160


/*    212 */	sxar2
/*    212 */	ldd,s	[%xg15+16],%f86
/*    212 */	ldd,s	[%xg14],%f92


/*    212 */	sxar2
/*    212 */	ldd,s	[%xg16],%f100
/*    212 */	fnmsubd,sc	%f36,%f158,%f32,%f34


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f36,%f158,%f38,%f40
/*    212 */	ldd,s	[%xg14+16],%f106


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f292,%f158,%f42,%f44
/*    212 */	fnmsubd,sc	%f292,%f158,%f46,%f36


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f52,%f158,%f50,%f48
/*    212 */	fnmsubd,sc	%f52,%f158,%f56,%f54


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f60,%f158,%f32,%f58
/*    212 */	fnmsubd,sc	%f60,%f158,%f38,%f62


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f316,%f158,%f42,%f64
/*    212 */	fnmsubd,sc	%f316,%f158,%f46,%f60


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f68,%f158,%f50,%f66
/*    212 */	fnmsubd,sc	%f68,%f158,%f56,%f70


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f34,%f34,%f72,%f34
/*    212 */	fmaddd,s	%f40,%f40,%f74,%f40


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f78,%f158,%f32,%f76
/*    212 */	fnmsubd,sc	%f78,%f158,%f38,%f80


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f334,%f158,%f42,%f82
/*    212 */	fnmsubd,sc	%f334,%f158,%f46,%f78


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f58,%f58,%f72,%f58
/*    212 */	fmaddd,s	%f62,%f62,%f74,%f62


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f86,%f158,%f50,%f84
/*    212 */	fnmsubd,sc	%f86,%f158,%f56,%f88


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f92,%f158,%f32,%f90
/*    212 */	fnmsubd,sc	%f92,%f158,%f38,%f94


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f44,%f44,%f34,%f44
/*    212 */	fmaddd,s	%f36,%f36,%f40,%f36


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f76,%f76,%f72,%f76
/*    212 */	fmaddd,s	%f80,%f80,%f74,%f80


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f348,%f158,%f42,%f96
/*    212 */	fnmsubd,sc	%f348,%f158,%f46,%f92


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f64,%f64,%f58,%f64
/*    212 */	fmaddd,s	%f60,%f60,%f62,%f60


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f100,%f158,%f32,%f98
/*    212 */	fnmsubd,sc	%f100,%f158,%f38,%f102


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f90,%f90,%f72,%f90
/*    212 */	fmaddd,s	%f94,%f94,%f74,%f94


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f48,%f48,%f44,%f48
/*    212 */	fmaddd,s	%f54,%f54,%f36,%f54


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f82,%f82,%f76,%f82
/*    212 */	fmaddd,s	%f78,%f78,%f80,%f78


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f106,%f158,%f50,%f104
/*    212 */	fnmsubd,sc	%f106,%f158,%f56,%f108


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f66,%f66,%f64,%f66
/*    212 */	fmaddd,s	%f70,%f70,%f60,%f70


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f96,%f96,%f90,%f96
/*    212 */	fmaddd,s	%f92,%f92,%f94,%f92


/*    212 */	sxar2
/*    212 */	frsqrtad,s	%f48,%f110
/*    212 */	frsqrtad,s	%f54,%f112


/*    212 */	sxar2
/*    212 */	fmuld,s	%f48,%f160,%f114
/*    212 */	fmuld,s	%f54,%f160,%f116


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f84,%f84,%f82,%f84
/*    212 */	fmaddd,s	%f88,%f88,%f78,%f88


/*    212 */	sxar2
/*    212 */	frsqrtad,s	%f66,%f118
/*    212 */	frsqrtad,s	%f70,%f120


/*    212 */	sxar2
/*    212 */	fmuld,s	%f66,%f160,%f122
/*    212 */	fmuld,s	%f70,%f160,%f124


/*    212 */	sxar2
/*    212 */	fmuld,s	%f110,%f110,%f126
/*    212 */	fmuld,s	%f112,%f112,%f128


/*    212 */	sxar2
/*    212 */	frsqrtad,s	%f84,%f130
/*    212 */	frsqrtad,s	%f88,%f132


/*    212 */	sxar2
/*    212 */	fmuld,s	%f118,%f118,%f134
/*    212 */	fmuld,s	%f120,%f120,%f136


/*    212 */	sxar2
/*    212 */	fmuld,s	%f84,%f160,%f138
/*    212 */	fmuld,s	%f88,%f160,%f140


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f114,%f126,%f160,%f126
/*    212 */	fnmsubd,s	%f116,%f128,%f160,%f128


/*    212 */	sxar2
/*    212 */	fmuld,s	%f130,%f130,%f142
/*    212 */	fmuld,s	%f132,%f132,%f144


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f122,%f134,%f160,%f134
/*    212 */	fnmsubd,s	%f124,%f136,%f160,%f136


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f110,%f126,%f110,%f110
/*    212 */	fmaddd,s	%f112,%f128,%f112,%f112


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f118,%f134,%f118,%f118
/*    212 */	fmaddd,s	%f120,%f136,%f120,%f120


/*    212 */	sxar2
/*    212 */	fmuld,s	%f110,%f110,%f146
/*    212 */	fmuld,s	%f112,%f112,%f148


/*    212 */	sxar2
/*    212 */	fmuld,s	%f118,%f118,%f150
/*    212 */	fmuld,s	%f120,%f120,%f152


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f114,%f146,%f160,%f146
/*    212 */	fnmsubd,s	%f116,%f148,%f160,%f148


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f122,%f150,%f160,%f150
/*    212 */	fnmsubd,s	%f124,%f152,%f160,%f152


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f110,%f146,%f110,%f110
/*    212 */	fmaddd,s	%f112,%f148,%f112,%f112


/*    212 */	sxar2
/*    212 */	fmuld,s	%f110,%f110,%f154
/*    212 */	fmuld,s	%f112,%f112,%f156

.L5815:


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f356,%f158,%f42,%f36
/*    212 */	fnmsubd,sc	%f356,%f158,%f46,%f100


/*    212 */	sxar2
/*    212 */	ldd,s	[%xg16+16],%f164
/*    212 */	fmaddd,s	%f98,%f98,%f72,%f98


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f104,%f104,%f96,%f104
/*    212 */	fmaddd,s	%f102,%f102,%f74,%f102


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f108,%f108,%f92,%f108
/*    212 */	fmaddd,s	%f118,%f150,%f118,%f118


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f120,%f152,%f120,%f120
/*    212 */	fnmsubd,s	%f138,%f142,%f160,%f142


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f114,%f154,%f160,%f114
/*    212 */	fnmsubd,s	%f140,%f144,%f160,%f144


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f116,%f156,%f160,%f116
/*    212 */	fnmsubd,sc	%f164,%f158,%f50,%f162


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f164,%f158,%f56,%f166
/*    212 */	fmaddd,s	%f36,%f36,%f98,%f36


/*    212 */	sxar2
/*    212 */	frsqrtad,s	%f104,%f168
/*    212 */	fmaddd,s	%f100,%f100,%f102,%f100


/*    212 */	sxar2
/*    212 */	frsqrtad,s	%f108,%f170
/*    212 */	add	%xg10,64,%xg20


/*    212 */	sxar2
/*    212 */	fmuld,s	%f118,%f118,%f172
/*    212 */	fmuld,s	%f120,%f120,%f174


/*    212 */	sxar2
/*    212 */	add	%xg12,%xg20,%xg21
/*    212 */	fmaddd,s	%f130,%f142,%f130,%f130


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f110,%f114,%f110,%f110
/*    212 */	ldd,s	[%xg21],%f190


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f132,%f144,%f132,%f132
/*    212 */	fmaddd,s	%f112,%f116,%f112,%f112


/*    212 */	sxar2
/*    212 */	fcmplted,s	%f72,%f48,%f48
/*    212 */	fcmplted,s	%f74,%f54,%f54


/*    212 */	sxar2
/*    212 */	fmuld,s	%f104,%f160,%f176
/*    212 */	fmuld,s	%f168,%f168,%f178


/*    212 */	sxar2
/*    212 */	fmuld,s	%f108,%f160,%f180
/*    212 */	fmuld,s	%f170,%f170,%f182


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f122,%f172,%f160,%f122
/*    212 */	fnmsubd,s	%f124,%f174,%f160,%f124


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f190,%f158,%f32,%f188
/*    212 */	fnmsubd,sc	%f190,%f158,%f38,%f192


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f162,%f162,%f36,%f162
/*    212 */	fmuld,s	%f130,%f130,%f194


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f166,%f166,%f100,%f166
/*    212 */	fmuld,s	%f132,%f132,%f196


/*    212 */	sxar2
/*    212 */	add	%xg4,%xg10,%xg22
/*    212 */	fand,s	%f110,%f48,%f110


/*    212 */	sxar2
/*    212 */	fand,s	%f112,%f54,%f112
/*    212 */	ldd,s	[%xg22],%f204


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f176,%f178,%f160,%f178
/*    212 */	fnmsubd,s	%f180,%f182,%f160,%f182


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f118,%f122,%f118,%f118
/*    212 */	fmaddd,s	%f120,%f124,%f120,%f120


/*    212 */	sxar2
/*    212 */	fcmplted,s	%f72,%f66,%f66
/*    212 */	fcmplted,s	%f74,%f70,%f70


/*    212 */	sxar2
/*    212 */	frsqrtad,s	%f162,%f198
/*    212 */	fnmsubd,s	%f138,%f194,%f160,%f194


/*    212 */	sxar2
/*    212 */	fmuld,s	%f162,%f160,%f200
/*    212 */	fnmsubd,sc	%f204,%f158,%f32,%f202


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f140,%f196,%f160,%f196
/*    212 */	fnmsubd,sc	%f204,%f158,%f38,%f206


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f168,%f178,%f168,%f168
/*    212 */	fmaddd,s	%f170,%f182,%f170,%f170


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f308,%f110,%f184,%f110
/*    212 */	fnmsubd,sc	%f308,%f112,%f186,%f52


/*    212 */	sxar2
/*    212 */	fand,s	%f118,%f66,%f118
/*    212 */	fand,s	%f120,%f70,%f120


/*    212 */	sxar2
/*    212 */	fmuld,s	%f198,%f198,%f208
/*    212 */	fmaddd,s	%f130,%f194,%f130,%f130


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f460,%f158,%f42,%f210
/*    212 */	fmaddd,s	%f202,%f202,%f72,%f202


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f460,%f158,%f46,%f204
/*    212 */	ldd,s	[%xg22+16],%f186


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f206,%f206,%f74,%f206
/*    212 */	fmuld,s	%f168,%f168,%f212


/*    212 */	sxar2
/*    212 */	fmuld,s	%f170,%f170,%f214
/*    212 */	frsqrtad,s	%f166,%f216


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f132,%f196,%f132,%f132
/*    212 */	fnmsubd,sc	%f324,%f118,%f110,%f118


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f324,%f120,%f52,%f68
/*    212 */	fmuld,s	%f130,%f130,%f218


/*    212 */	sxar2
/*    212 */	fmuld,s	%f166,%f160,%f220
/*    212 */	fnmsubd,sc	%f186,%f158,%f50,%f222


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f210,%f210,%f202,%f210
/*    212 */	fnmsubd,sc	%f186,%f158,%f56,%f224


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f204,%f204,%f206,%f204
/*    212 */	fnmsubd,s	%f176,%f212,%f160,%f212


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f180,%f214,%f160,%f214
/*    212 */	fmuld,s	%f216,%f216,%f226


/*    212 */	sxar2
/*    212 */	fmuld,s	%f132,%f132,%f228
/*    212 */	fnmsubd,sc	%f446,%f158,%f42,%f230


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f446,%f158,%f46,%f190
/*    212 */	ldd,s	[%xg21+16],%f52


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f188,%f188,%f72,%f188
/*    212 */	fmaddd,s	%f222,%f222,%f210,%f222


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f192,%f192,%f74,%f192
/*    212 */	fmaddd,s	%f224,%f224,%f204,%f224


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f168,%f212,%f168,%f168
/*    212 */	fmaddd,s	%f170,%f214,%f170,%f170


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f200,%f208,%f160,%f208
/*    212 */	fnmsubd,s	%f138,%f218,%f160,%f138


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f220,%f226,%f160,%f226
/*    212 */	fnmsubd,s	%f140,%f228,%f160,%f140


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f52,%f158,%f50,%f48
/*    212 */	fnmsubd,sc	%f52,%f158,%f56,%f54


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f230,%f230,%f188,%f230
/*    212 */	frsqrtad,s	%f222,%f184


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f190,%f190,%f192,%f190
/*    212 */	frsqrtad,s	%f224,%f232


/*    212 */	sxar2
/*    212 */	add	%xg10,128,%xg23
/*    212 */	fmuld,s	%f168,%f168,%f234


/*    212 */	sxar2
/*    212 */	fmuld,s	%f170,%f170,%f236
/*    212 */	add	%xg12,%xg23,%xg24


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f198,%f208,%f198,%f198
/*    212 */	fmaddd,s	%f130,%f138,%f130,%f130


/*    212 */	sxar2
/*    212 */	ldd,s	[%xg24],%f248
/*    212 */	fmaddd,s	%f216,%f226,%f216,%f216


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f132,%f140,%f132,%f132
/*    212 */	fcmplted,s	%f72,%f84,%f84


/*    212 */	sxar2
/*    212 */	fcmplted,s	%f74,%f88,%f88
/*    212 */	fmuld,s	%f222,%f160,%f238


/*    212 */	sxar2
/*    212 */	fmuld,s	%f184,%f184,%f240
/*    212 */	fmuld,s	%f224,%f160,%f242


/*    212 */	sxar2
/*    212 */	fmuld,s	%f232,%f232,%f244
/*    212 */	fnmsubd,s	%f176,%f234,%f160,%f176


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f180,%f236,%f160,%f180
/*    212 */	fnmsubd,sc	%f248,%f158,%f32,%f246


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f248,%f158,%f38,%f250
/*    212 */	fmaddd,s	%f48,%f48,%f230,%f48


/*    212 */	sxar2
/*    212 */	fmuld,s	%f198,%f198,%f252
/*    212 */	fmaddd,s	%f54,%f54,%f190,%f54


/*    212 */	sxar2
/*    212 */	fmuld,s	%f216,%f216,%f254
/*    212 */	add	%xg4,%xg20,%xg20


/*    212 */	sxar2
/*    212 */	fand,s	%f130,%f84,%f130
/*    212 */	fand,s	%f132,%f88,%f132


/*    212 */	sxar2
/*    212 */	ldd,s	[%xg20],%f36
/*    212 */	fnmsubd,s	%f238,%f240,%f160,%f240


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f242,%f244,%f160,%f244
/*    212 */	fmaddd,s	%f168,%f176,%f168,%f168


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f170,%f180,%f170,%f170
/*    212 */	fcmplted,s	%f72,%f104,%f104


/*    212 */	sxar2
/*    212 */	fcmplted,s	%f74,%f108,%f108
/*    212 */	frsqrtad,s	%f48,%f110


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f200,%f252,%f160,%f252
/*    212 */	fmuld,s	%f48,%f160,%f114


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f36,%f158,%f32,%f34
/*    212 */	fnmsubd,s	%f220,%f254,%f160,%f254


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f36,%f158,%f38,%f40
/*    212 */	fmaddd,s	%f184,%f240,%f184,%f184


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f232,%f244,%f232,%f232
/*    212 */	fnmsubd,sc	%f342,%f130,%f118,%f130


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f342,%f132,%f68,%f86
/*    212 */	fand,s	%f168,%f104,%f168


/*    212 */	sxar2
/*    212 */	fand,s	%f170,%f108,%f170
/*    212 */	fmuld,s	%f110,%f110,%f44


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f198,%f252,%f198,%f198
/*    212 */	fnmsubd,sc	%f292,%f158,%f42,%f58


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f34,%f34,%f72,%f34
/*    212 */	fnmsubd,sc	%f292,%f158,%f46,%f36


/*    212 */	sxar2
/*    212 */	ldd,s	[%xg20+16],%f68
/*    212 */	fmaddd,s	%f40,%f40,%f74,%f40


/*    212 */	sxar2
/*    212 */	fmuld,s	%f184,%f184,%f60
/*    212 */	fmuld,s	%f232,%f232,%f62


/*    212 */	sxar2
/*    212 */	frsqrtad,s	%f54,%f112
/*    212 */	fmaddd,s	%f216,%f254,%f216,%f216


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f362,%f168,%f130,%f168
/*    212 */	fnmsubd,sc	%f362,%f170,%f86,%f106


/*    212 */	sxar2
/*    212 */	fmuld,s	%f198,%f198,%f64
/*    212 */	fmuld,s	%f54,%f160,%f116


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f68,%f158,%f50,%f66
/*    212 */	fmaddd,s	%f58,%f58,%f34,%f58


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f68,%f158,%f56,%f70
/*    212 */	fmaddd,s	%f36,%f36,%f40,%f36


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f238,%f60,%f160,%f60
/*    212 */	fnmsubd,s	%f242,%f62,%f160,%f62


/*    212 */	sxar2
/*    212 */	fmuld,s	%f112,%f112,%f76
/*    212 */	fmuld,s	%f216,%f216,%f78


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f504,%f158,%f42,%f80
/*    212 */	fnmsubd,sc	%f504,%f158,%f46,%f248


/*    212 */	sxar2
/*    212 */	ldd,s	[%xg24+16],%f86
/*    212 */	fmaddd,s	%f246,%f246,%f72,%f246


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f66,%f66,%f58,%f66
/*    212 */	fmaddd,s	%f250,%f250,%f74,%f250


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f70,%f70,%f36,%f70
/*    212 */	fmaddd,s	%f184,%f60,%f184,%f184


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f232,%f62,%f232,%f232
/*    212 */	fnmsubd,s	%f114,%f44,%f160,%f44


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f200,%f64,%f160,%f200
/*    212 */	fnmsubd,s	%f116,%f76,%f160,%f76


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f220,%f78,%f160,%f220
/*    212 */	fnmsubd,sc	%f86,%f158,%f50,%f84


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f86,%f158,%f56,%f88
/*    212 */	fmaddd,s	%f80,%f80,%f246,%f80


/*    212 */	sxar2
/*    212 */	frsqrtad,s	%f66,%f118
/*    212 */	fmaddd,s	%f248,%f248,%f250,%f248


/*    212 */	sxar2
/*    212 */	frsqrtad,s	%f70,%f120
/*    212 */	add	%xg10,192,%xg10


/*    212 */	sxar2
/*    212 */	fmuld,s	%f184,%f184,%f82
/*    212 */	fmuld,s	%f232,%f232,%f90


/*    212 */	sxar2
/*    212 */	add	%xg12,%xg10,%xg16
/*    212 */	fmaddd,s	%f110,%f44,%f110,%f110


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f198,%f200,%f198,%f198
/*    212 */	ldd,s	[%xg16],%f100


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f112,%f76,%f112,%f112
/*    212 */	fmaddd,s	%f216,%f220,%f216,%f216


/*    212 */	sxar2
/*    212 */	fcmplted,s	%f72,%f162,%f162
/*    212 */	fcmplted,s	%f74,%f166,%f166


/*    212 */	sxar2
/*    212 */	fmuld,s	%f66,%f160,%f122
/*    212 */	fmuld,s	%f118,%f118,%f94


/*    212 */	sxar2
/*    212 */	fmuld,s	%f70,%f160,%f124
/*    212 */	fmuld,s	%f120,%f120,%f96


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f238,%f82,%f160,%f238
/*    212 */	fnmsubd,s	%f242,%f90,%f160,%f242


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f100,%f158,%f32,%f98
/*    212 */	fnmsubd,sc	%f100,%f158,%f38,%f102


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f84,%f84,%f80,%f84
/*    212 */	fmuld,s	%f110,%f110,%f104


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f88,%f88,%f248,%f88
/*    212 */	fmuld,s	%f112,%f112,%f108


/*    212 */	sxar2
/*    212 */	add	%xg4,%xg23,%xg23
/*    212 */	fand,s	%f198,%f162,%f198


/*    212 */	sxar2
/*    212 */	fand,s	%f216,%f166,%f216
/*    212 */	ldd,s	[%xg23],%f92


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f122,%f94,%f160,%f94
/*    212 */	fnmsubd,s	%f124,%f96,%f160,%f96


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f184,%f238,%f184,%f184
/*    212 */	fmaddd,s	%f232,%f242,%f232,%f232


/*    212 */	sxar2
/*    212 */	fcmplted,s	%f72,%f222,%f222
/*    212 */	fcmplted,s	%f74,%f224,%f224


/*    212 */	sxar2
/*    212 */	frsqrtad,s	%f84,%f130
/*    212 */	fnmsubd,s	%f114,%f104,%f160,%f104


/*    212 */	sxar2
/*    212 */	fmuld,s	%f84,%f160,%f138
/*    212 */	fnmsubd,sc	%f92,%f158,%f32,%f126


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f116,%f108,%f160,%f108
/*    212 */	fnmsubd,sc	%f92,%f158,%f38,%f128


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f118,%f94,%f118,%f118
/*    212 */	fmaddd,s	%f120,%f96,%f120,%f120


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f420,%f198,%f168,%f198
/*    212 */	fnmsubd,sc	%f420,%f216,%f106,%f164


/*    212 */	sxar2
/*    212 */	fand,s	%f184,%f222,%f184
/*    212 */	fand,s	%f232,%f224,%f232


/*    212 */	sxar2
/*    212 */	fmuld,s	%f130,%f130,%f142
/*    212 */	fmaddd,s	%f110,%f104,%f110,%f110


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f348,%f158,%f42,%f96
/*    212 */	fmaddd,s	%f126,%f126,%f72,%f126


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f348,%f158,%f46,%f92
/*    212 */	ldd,s	[%xg23+16],%f106


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f128,%f128,%f74,%f128
/*    212 */	fmuld,s	%f118,%f118,%f150


/*    212 */	sxar2
/*    212 */	fmuld,s	%f120,%f120,%f152
/*    212 */	frsqrtad,s	%f88,%f132


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f112,%f108,%f112,%f112
/*    212 */	fnmsubd,sc	%f442,%f184,%f198,%f184


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f442,%f232,%f164,%f186
/*    212 */	fmuld,s	%f110,%f110,%f154


/*    212 */	sxar2
/*    212 */	fmuld,s	%f88,%f160,%f140
/*    212 */	fnmsubd,sc	%f106,%f158,%f50,%f104


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f96,%f96,%f126,%f96
/*    212 */	fnmsubd,sc	%f106,%f158,%f56,%f108


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f92,%f92,%f128,%f92
/*    212 */	fnmsubd,s	%f122,%f150,%f160,%f150


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f124,%f152,%f160,%f152
/*    212 */	fmuld,s	%f132,%f132,%f144


/*    212 */	sxar2
/*    212 */	fmuld,s	%f112,%f112,%f156
/*    212 */	sub	%xg8,6,%xg8

/*    212 */	sxar1
/*    212 */	cmp	%xg8,15

/*    212 */	bge,pt	%icc, .L5815
	nop


.L5931:


/*    212 */	sxar2
/*    ??? */	ldd,s	[%fp+255],%f248
/*    212 */	ldd,s	[%xg16+16],%f162


/*    212 */	sxar2
/*    212 */	add	%xg4,%xg10,%xg17
/*    212 */	fmaddd,s	%f98,%f98,%f72,%f98


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f104,%f104,%f96,%f104
/*    212 */	ldd,s	[%xg17],%f168


/*    212 */	sxar2
/*    212 */	ldd,s	[%xg17+16],%f188
/*    212 */	fmaddd,s	%f102,%f102,%f74,%f102


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f108,%f108,%f92,%f108
/*    212 */	add	%xg10,64,%xg10


/*    212 */	sxar2
/*    ??? */	ldd,s	[%fp+239],%f236
/*    212 */	fcmplted,s	%f72,%f48,%f48


/*    212 */	sxar2
/*    212 */	sub	%xg8,6,%xg8
/*    212 */	fmaddd,s	%f118,%f150,%f118,%f118


/*    212 */	sxar2
/*    212 */	fcmplted,s	%f74,%f54,%f54
/*    212 */	fnmsubd,sc	%f356,%f248,%f42,%f158


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f356,%f248,%f46,%f100
/*    212 */	fnmsubd,sc	%f162,%f248,%f50,%f160


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f168,%f248,%f32,%f166
/*    212 */	fnmsubd,sc	%f168,%f248,%f38,%f170


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f424,%f248,%f42,%f172
/*    212 */	fnmsubd,sc	%f162,%f248,%f56,%f164


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f424,%f248,%f46,%f168
/*    212 */	fnmsubd,s	%f138,%f142,%f236,%f142


/*    212 */	sxar2
/*    212 */	frsqrtad,s	%f104,%f174
/*    212 */	frsqrtad,s	%f108,%f176


/*    212 */	sxar2
/*    212 */	fmuld,s	%f104,%f236,%f178
/*    212 */	fmaddd,s	%f158,%f158,%f98,%f158


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f100,%f100,%f102,%f100
/*    212 */	fnmsubd,sc	%f188,%f248,%f50,%f182


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f166,%f166,%f72,%f166
/*    212 */	fnmsubd,sc	%f188,%f248,%f56,%f190


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f170,%f170,%f74,%f170
/*    212 */	fnmsubd,s	%f140,%f144,%f236,%f144


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f114,%f154,%f236,%f114
/*    212 */	fmuld,s	%f108,%f236,%f180


/*    212 */	sxar2
/*    212 */	fmuld,s	%f174,%f174,%f192
/*    212 */	fmaddd,s	%f130,%f142,%f130,%f130


/*    212 */	sxar2
/*    212 */	fmuld,s	%f176,%f176,%f194
/*    212 */	fmaddd,s	%f160,%f160,%f158,%f160


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f164,%f164,%f100,%f164
/*    212 */	fnmsubd,s	%f116,%f156,%f236,%f116


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f172,%f172,%f166,%f172
/*    212 */	fmaddd,s	%f120,%f152,%f120,%f120


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f168,%f168,%f170,%f168
/*    212 */	fmaddd,s	%f132,%f144,%f132,%f132


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f110,%f114,%f110,%f110
/*    212 */	fnmsubd,s	%f178,%f192,%f236,%f192


/*    212 */	sxar2
/*    212 */	fmuld,s	%f118,%f118,%f208
/*    212 */	fmuld,s	%f130,%f130,%f196


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f180,%f194,%f236,%f194
/*    212 */	frsqrtad,s	%f160,%f200


/*    212 */	sxar2
/*    212 */	frsqrtad,s	%f164,%f202
/*    212 */	fmuld,s	%f160,%f236,%f204


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f182,%f182,%f172,%f182
/*    212 */	fmuld,s	%f164,%f236,%f206


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f190,%f190,%f168,%f190
/*    212 */	fmaddd,s	%f112,%f116,%f112,%f112


/*    212 */	sxar2
/*    212 */	fand,s	%f110,%f48,%f110
/*    212 */	fmuld,s	%f132,%f132,%f198


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f174,%f192,%f174,%f174
/*    212 */	fnmsubd,s	%f138,%f196,%f236,%f196


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f176,%f194,%f176,%f176
/*    212 */	fmuld,s	%f200,%f200,%f212


/*    212 */	sxar2
/*    212 */	fmuld,s	%f202,%f202,%f214
/*    212 */	frsqrtad,s	%f182,%f216


/*    212 */	sxar2
/*    212 */	fmuld,s	%f182,%f236,%f220
/*    212 */	frsqrtad,s	%f190,%f218


/*    212 */	sxar2
/*    212 */	fmuld,s	%f190,%f236,%f222
/*    212 */	fand,s	%f112,%f54,%f112


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f308,%f110,%f184,%f110
/*    212 */	fnmsubd,s	%f140,%f198,%f236,%f198


/*    212 */	sxar2
/*    212 */	fmuld,s	%f174,%f174,%f224
/*    212 */	fmuld,s	%f176,%f176,%f226


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f130,%f196,%f130,%f130
/*    212 */	fnmsubd,s	%f204,%f212,%f236,%f212


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f206,%f214,%f236,%f214
/*    212 */	fmuld,s	%f120,%f120,%f210


/*    212 */	sxar2
/*    212 */	fmuld,s	%f216,%f216,%f228
/*    212 */	fnmsubd,s	%f122,%f208,%f236,%f122


/*    212 */	sxar2
/*    212 */	fmuld,s	%f218,%f218,%f230
/*    212 */	fcmplted,s	%f72,%f66,%f66


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f308,%f112,%f186,%f52
/*    212 */	fmaddd,s	%f132,%f198,%f132,%f132


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f178,%f224,%f236,%f224
/*    212 */	fnmsubd,s	%f180,%f226,%f236,%f226


/*    212 */	sxar2
/*    212 */	fmuld,s	%f130,%f130,%f232
/*    212 */	fmaddd,s	%f200,%f212,%f200,%f200


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f202,%f214,%f202,%f202
/*    212 */	fnmsubd,s	%f124,%f210,%f236,%f124


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f220,%f228,%f236,%f228
/*    212 */	fmaddd,s	%f118,%f122,%f118,%f118


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f222,%f230,%f236,%f230
/*    212 */	fcmplted,s	%f74,%f70,%f70


/*    212 */	sxar2
/*    212 */	fcmplted,s	%f72,%f84,%f84
/*    212 */	fmuld,s	%f132,%f132,%f234


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f174,%f224,%f174,%f174
/*    212 */	fmaddd,s	%f176,%f226,%f176,%f176


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f138,%f232,%f236,%f138
/*    212 */	fmuld,s	%f200,%f200,%f238


/*    212 */	sxar2
/*    212 */	fmuld,s	%f202,%f202,%f240
/*    212 */	fmaddd,s	%f120,%f124,%f120,%f120


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f216,%f228,%f216,%f184
/*    212 */	fand,s	%f118,%f66,%f118


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f218,%f230,%f218,%f186
/*    212 */	fcmplted,s	%f74,%f88,%f88


/*    212 */	sxar2
/*    212 */	fcmplted,s	%f72,%f104,%f104
/*    212 */	fnmsubd,s	%f140,%f234,%f236,%f140


/*    212 */	sxar2
/*    212 */	fmuld,s	%f174,%f174,%f242
/*    212 */	fmuld,s	%f176,%f176,%f244


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f130,%f138,%f130,%f130
/*    212 */	fnmsubd,s	%f204,%f238,%f236,%f238


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f206,%f240,%f236,%f240
/*    212 */	fand,s	%f120,%f70,%f120


/*    212 */	sxar2
/*    212 */	fmuld,s	%f184,%f184,%f246
/*    212 */	fnmsubd,sc	%f324,%f118,%f110,%f118


/*    212 */	sxar2
/*    212 */	fmuld,s	%f186,%f186,%f248
/*    212 */	fcmplted,s	%f74,%f108,%f108


/*    212 */	sxar2
/*    212 */	fcmplted,s	%f72,%f160,%f160
/*    212 */	fmaddd,s	%f132,%f140,%f132,%f132


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f178,%f242,%f236,%f178
/*    212 */	fnmsubd,s	%f180,%f244,%f236,%f180


/*    212 */	sxar2
/*    212 */	fand,s	%f130,%f84,%f130
/*    212 */	fmaddd,s	%f200,%f238,%f200,%f200


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f202,%f240,%f202,%f202
/*    212 */	fnmsubd,sc	%f324,%f120,%f52,%f68


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f220,%f246,%f236,%f246
/*    212 */	fnmsubd,s	%f222,%f248,%f236,%f248


/*    212 */	sxar2
/*    212 */	fcmplted,s	%f74,%f164,%f164
/*    212 */	fcmplted,s	%f72,%f182,%f182


/*    212 */	sxar2
/*    212 */	fcmplted,s	%f74,%f190,%f190
/*    212 */	fand,s	%f132,%f88,%f132


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f174,%f178,%f174,%f174
/*    212 */	fmaddd,s	%f176,%f180,%f176,%f176


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f342,%f130,%f118,%f130
/*    212 */	fmuld,s	%f200,%f200,%f250


/*    212 */	sxar2
/*    212 */	fmuld,s	%f202,%f202,%f252
/*    212 */	fmaddd,s	%f184,%f246,%f184,%f184


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f186,%f248,%f186,%f186
/*    212 */	fnmsubd,sc	%f342,%f132,%f68,%f86


/*    212 */	sxar2
/*    212 */	fand,s	%f174,%f104,%f174
/*    212 */	fand,s	%f176,%f108,%f176


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f204,%f250,%f236,%f204
/*    212 */	fnmsubd,s	%f206,%f252,%f236,%f206


/*    212 */	sxar2
/*    212 */	fmuld,s	%f184,%f184,%f254
/*    212 */	fmuld,s	%f186,%f186,%f34


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f362,%f174,%f130,%f174
/*    212 */	fnmsubd,sc	%f362,%f176,%f86,%f106


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f200,%f204,%f200,%f200
/*    212 */	fmaddd,s	%f202,%f206,%f202,%f202


/*    212 */	sxar2
/*    212 */	fnmsubd,s	%f220,%f254,%f236,%f220
/*    212 */	fnmsubd,s	%f222,%f34,%f236,%f222


/*    212 */	sxar2
/*    212 */	fand,s	%f200,%f160,%f200
/*    212 */	fand,s	%f202,%f164,%f202


/*    212 */	sxar2
/*    212 */	fmaddd,s	%f184,%f220,%f184,%f184
/*    212 */	fmaddd,s	%f186,%f222,%f186,%f186


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f418,%f200,%f174,%f200
/*    212 */	fnmsubd,sc	%f418,%f202,%f106,%f162


/*    212 */	sxar2
/*    212 */	fand,s	%f184,%f182,%f184
/*    212 */	fand,s	%f186,%f190,%f186


/*    212 */	sxar2
/*    212 */	fnmsubd,sc	%f444,%f184,%f200,%f184
/*    212 */	fnmsubd,sc	%f444,%f186,%f162,%f186

.L5927:


.L5926:


.L5929:


/*    149 */	sxar2
/*    149 */	add	%xg12,%xg10,%xg25
/* #00004 */	ldd,s	[%fp+255],%f244


/*     22 */	sxar2
/*     22 */	add	%xg4,%xg10,%xg26
/*     22 */	ldd,s	[%xg25],%f132


/*    224 */	sxar2
/*    224 */	ldd,s	[%xg25+16],%f140
/*    224 */	add	%xg10,64,%xg10


/*     38 */	sxar2
/*     38 */	subcc	%xg8,2,%xg8
/* #00004 */	ldd,s	[%fp+239],%f246


/*    149 */	sxar2
/*    149 */	ldd,s	[%xg26],%f166
/*    149 */	ldd,s	[%xg26+16],%f174


/*    149 */	sxar2
/*    149 */	fnmsubd,sc	%f132,%f244,%f32,%f130
/*    149 */	fnmsubd,sc	%f132,%f244,%f38,%f134


/*    190 */	sxar2
/*    190 */	fnmsubd,sc	%f388,%f244,%f42,%f136
/*    190 */	fnmsubd,sc	%f388,%f244,%f46,%f132


/*    149 */	sxar2
/*    149 */	fnmsubd,sc	%f140,%f244,%f50,%f138
/*    149 */	fnmsubd,sc	%f140,%f244,%f56,%f142


/*    149 */	sxar2
/*    149 */	fnmsubd,sc	%f166,%f244,%f32,%f164
/*    149 */	fnmsubd,sc	%f166,%f244,%f38,%f168


/*    190 */	sxar2
/*    190 */	fnmsubd,sc	%f422,%f244,%f42,%f170
/*    190 */	fnmsubd,sc	%f422,%f244,%f46,%f166


/*    149 */	sxar2
/*    149 */	fnmsubd,sc	%f174,%f244,%f50,%f172
/*    149 */	fnmsubd,sc	%f174,%f244,%f56,%f176


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f130,%f130,%f72,%f130
/*     44 */	fmaddd,s	%f134,%f134,%f74,%f134


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f164,%f164,%f72,%f164
/*     44 */	fmaddd,s	%f168,%f168,%f74,%f168


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f136,%f136,%f130,%f136
/*     44 */	fmaddd,s	%f132,%f132,%f134,%f132


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f170,%f170,%f164,%f170
/*     44 */	fmaddd,s	%f166,%f166,%f168,%f166


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f138,%f138,%f136,%f138
/*     44 */	fmaddd,s	%f142,%f142,%f132,%f142


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f172,%f172,%f170,%f172
/*     44 */	fmaddd,s	%f176,%f176,%f166,%f176


/*     38 */	sxar2
/*     38 */	frsqrtad,s	%f138,%f144
/*     38 */	fmuld,s	%f138,%f246,%f146


/*     38 */	sxar2
/*     38 */	frsqrtad,s	%f142,%f152
/*     38 */	fmuld,s	%f142,%f246,%f154


/*    110 */	sxar2
/*    110 */	fcmplted,s	%f72,%f138,%f138
/*    110 */	fcmplted,s	%f74,%f142,%f142


/*     38 */	sxar2
/*     38 */	fmuld,s	%f172,%f246,%f178
/*     38 */	fmuld,s	%f176,%f246,%f188


/*     32 */	sxar2
/*     32 */	fmuld,s	%f144,%f144,%f148
/*     32 */	fmuld,s	%f152,%f152,%f158


/*     32 */	sxar2
/*     32 */	fnmsubd,s	%f146,%f148,%f246,%f148
/*     32 */	fnmsubd,s	%f154,%f158,%f246,%f158


/*     32 */	sxar2
/*     32 */	fmaddd,s	%f144,%f148,%f144,%f144
/*     32 */	fmaddd,s	%f152,%f158,%f152,%f152


/*     32 */	sxar2
/*     32 */	fmuld,s	%f144,%f144,%f150
/*     32 */	fmuld,s	%f152,%f152,%f160


/*     32 */	sxar2
/*     32 */	fnmsubd,s	%f146,%f150,%f246,%f150
/*     32 */	fnmsubd,s	%f154,%f160,%f246,%f160


/*     32 */	sxar2
/*     32 */	fmaddd,s	%f144,%f150,%f144,%f144
/*     32 */	fmaddd,s	%f152,%f160,%f152,%f152


/*     32 */	sxar2
/*     32 */	fmuld,s	%f144,%f144,%f156
/*     32 */	fmuld,s	%f152,%f152,%f162


/*     32 */	sxar2
/*     32 */	fnmsubd,s	%f146,%f156,%f246,%f146
/*     32 */	fnmsubd,s	%f154,%f162,%f246,%f154


/*     32 */	sxar2
/*     32 */	fmaddd,s	%f144,%f146,%f144,%f144
/*     32 */	fmaddd,s	%f152,%f154,%f152,%f152


/*    115 */	sxar2
/*    115 */	fand,s	%f144,%f138,%f144
/*    115 */	fand,s	%f152,%f142,%f152


/*     60 */	sxar2
/*     60 */	fnmsubd,sc	%f396,%f144,%f184,%f144
/*     60 */	frsqrtad,s	%f172,%f184


/*     60 */	sxar2
/*     60 */	fnmsubd,sc	%f396,%f152,%f186,%f140
/*     60 */	frsqrtad,s	%f176,%f186


/*    110 */	sxar2
/*    110 */	fcmplted,s	%f72,%f172,%f172
/*    110 */	fcmplted,s	%f74,%f176,%f176


/*     32 */	sxar2
/*     32 */	fmuld,s	%f184,%f184,%f180
/*     32 */	fmuld,s	%f186,%f186,%f190


/*     32 */	sxar2
/*     32 */	fnmsubd,s	%f178,%f180,%f246,%f180
/*     32 */	fnmsubd,s	%f188,%f190,%f246,%f190


/*     32 */	sxar2
/*     32 */	fmaddd,s	%f184,%f180,%f184,%f184
/*     32 */	fmaddd,s	%f186,%f190,%f186,%f186


/*     32 */	sxar2
/*     32 */	fmuld,s	%f184,%f184,%f182
/*     32 */	fmuld,s	%f186,%f186,%f194


/*     32 */	sxar2
/*     32 */	fnmsubd,s	%f178,%f182,%f246,%f182
/*     32 */	fnmsubd,s	%f188,%f194,%f246,%f194


/*     32 */	sxar2
/*     32 */	fmaddd,s	%f184,%f182,%f184,%f184
/*     32 */	fmaddd,s	%f186,%f194,%f186,%f186


/*     32 */	sxar2
/*     32 */	fmuld,s	%f184,%f184,%f192
/*     32 */	fmuld,s	%f186,%f186,%f196


/*     32 */	sxar2
/*     32 */	fnmsubd,s	%f178,%f192,%f246,%f178
/*     32 */	fnmsubd,s	%f188,%f196,%f246,%f188


/*     32 */	sxar2
/*     32 */	fmaddd,s	%f184,%f178,%f184,%f184
/*     32 */	fmaddd,s	%f186,%f188,%f186,%f186


/*    115 */	sxar2
/*    115 */	fand,s	%f184,%f172,%f184
/*    115 */	fand,s	%f186,%f176,%f186


/*    223 */	sxar2
/*    223 */	fnmsubd,sc	%f430,%f184,%f144,%f184
/*    223 */	fnmsubd,sc	%f430,%f186,%f140,%f186

/*    224 */	bpos,pt	%icc, .L5929
	nop


.L5925:


.L5823:

/*    224 */	sxar1
/*    224 */	addcc	%xg8,1,%xg8

/*    224 */	bneg	.L5816
	nop


.L5824:

/*    224 */	sxar1
/*    224 */	ldx	[%i0+2199],%xg28

.L5829:


/*    149 */	sxar2
/*    149 */	add	%xg28,%xg10,%xg27
/* #00003 */	ldd,s	[%fp+255],%f240


/*     22 */	sxar2
/*     22 */	add	%xg10,32,%xg10
/*     22 */	ldd,s	[%xg27],%f200


/*    224 */	sxar2
/*    224 */	ldd,s	[%xg27+16],%f208
/*    224 */	subcc	%xg8,1,%xg8


/*    149 */	sxar2
/* #00003 */	ldd,s	[%fp+239],%f242
/*    149 */	fnmsubd,sc	%f200,%f240,%f32,%f198


/*    190 */	sxar2
/*    190 */	fnmsubd,sc	%f200,%f240,%f38,%f202
/*    190 */	fnmsubd,sc	%f456,%f240,%f42,%f204


/*    149 */	sxar2
/*    149 */	fnmsubd,sc	%f456,%f240,%f46,%f200
/*    149 */	fnmsubd,sc	%f208,%f240,%f50,%f206


/*     44 */	sxar2
/*     44 */	fnmsubd,sc	%f208,%f240,%f56,%f210
/*     44 */	fmaddd,s	%f198,%f198,%f72,%f198


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f202,%f202,%f74,%f202
/*     44 */	fmaddd,s	%f204,%f204,%f198,%f204


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f200,%f200,%f202,%f200
/*     44 */	fmaddd,s	%f206,%f206,%f204,%f206


/*     60 */	sxar2
/*     60 */	fmaddd,s	%f210,%f210,%f200,%f210
/*     60 */	frsqrtad,s	%f206,%f212


/*     38 */	sxar2
/*     38 */	frsqrtad,s	%f210,%f220
/*     38 */	fmuld,s	%f206,%f242,%f214


/*    110 */	sxar2
/*    110 */	fmuld,s	%f210,%f242,%f222
/*    110 */	fcmplted,s	%f72,%f206,%f206


/*     32 */	sxar2
/*     32 */	fcmplted,s	%f74,%f210,%f210
/*     32 */	fmuld,s	%f212,%f212,%f216


/*     32 */	sxar2
/*     32 */	fmuld,s	%f220,%f220,%f226
/*     32 */	fnmsubd,s	%f214,%f216,%f242,%f216


/*     32 */	sxar2
/*     32 */	fnmsubd,s	%f222,%f226,%f242,%f226
/*     32 */	fmaddd,s	%f212,%f216,%f212,%f212


/*     32 */	sxar2
/*     32 */	fmaddd,s	%f220,%f226,%f220,%f220
/*     32 */	fmuld,s	%f212,%f212,%f218


/*     32 */	sxar2
/*     32 */	fmuld,s	%f220,%f220,%f228
/*     32 */	fnmsubd,s	%f214,%f218,%f242,%f218


/*     32 */	sxar2
/*     32 */	fnmsubd,s	%f222,%f228,%f242,%f228
/*     32 */	fmaddd,s	%f212,%f218,%f212,%f212


/*     32 */	sxar2
/*     32 */	fmaddd,s	%f220,%f228,%f220,%f220
/*     32 */	fmuld,s	%f212,%f212,%f224


/*     32 */	sxar2
/*     32 */	fmuld,s	%f220,%f220,%f230
/*     32 */	fnmsubd,s	%f214,%f224,%f242,%f214


/*     32 */	sxar2
/*     32 */	fnmsubd,s	%f222,%f230,%f242,%f222
/*     32 */	fmaddd,s	%f212,%f214,%f212,%f212


/*    115 */	sxar2
/*    115 */	fmaddd,s	%f220,%f222,%f220,%f220
/*    115 */	fand,s	%f212,%f206,%f212


/*    222 */	sxar2
/*    222 */	fand,s	%f220,%f210,%f220
/*    222 */	fnmsubd,sc	%f464,%f212,%f184,%f184

/*    223 */	sxar1
/*    223 */	fnmsubd,sc	%f464,%f220,%f186,%f186

/*    224 */	bpos,pt	%icc, .L5829
	nop


.L5825:


.L5816:


/*    224 */	sxar2
/*    224 */	std,s	%f184,[%fp+1263]
/*    224 */	std,s	%f186,[%fp+1279]

.L5817:


/*     22 */	sxar2
/*     22 */	add	%xg19,%xg1,%xg18
/*     22 */	ldd,s	[%fp+1263],%f236



/*    226 */	sxar2
/*    226 */	add	%xg1,32,%xg1
/*    226 */	add	%xg3,128,%xg3


/*     24 */	sxar2
/*     24 */	subcc	%xg0,1,%xg0
/*     24 */	std,s	%f236,[%xg18]


/*     25 */	sxar2
/*     25 */	ldd,s	[%fp+1279],%f238
/*     25 */	std,s	%f238,[%xg18+16]

/*    226 */	bne,pt	%icc, .L5811
/*    226 */	add	%o0,4,%o0


.L5818:

/*    226 */
/*    226 */	ba	.L5809
	nop


.L5820:

/*    226 *//*    226 */	call	__mpc_obar
/*    226 */	ldx	[%fp+2199],%o0

/*    226 *//*    226 */	call	__mpc_obar
/*    226 */	ldx	[%fp+2199],%o0


.L5821:

/*    226 */	ret
	restore



.LLFE11:
	.size	_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_5,.-_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_5
	.type	_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_5,#function
	.global	__gxx_personality_v0
	.section	".eh_frame",#alloc
.LLframe1:
	.uaword	.LLECIE1-.LLSCIE1	! CIE Length
.LLSCIE1:
	.uaword	0x0	! CIE ID
	.byte	0x1	! CIE Version
	.asciz	"zPLR"	! CIE Augmentation
	.uleb128	0x1	! CIE Code Alignment Factor
	.sleb128	-8	! CIE Data Alignment Factor
	.byte	0xf
	.uleb128	0xb	! CIE Augmentation Section Length 
	.byte	0x0	! Personality Routine Encoding Specifier ( absptr )
	.uaxword	__gxx_personality_v0	! Personality Routine Name
	.byte	0x1b	! LSDA Encoding Specifier ( pcrel | sdata4 )
	.byte	0x1b	! FDE Code Encoding Specifier ( pcrel | sdata4 )
	.byte	0xc	! DW_CFA_def_cfa
	.uleb128	0xe
	.uleb128	0x7ff
	.align	8	! CIE Padding
.LLECIE1:
.LLSFDE1:
	.uaword	.LLEFDE1-.LLASFDE1	! FDE Length
.LLASFDE1:
	.uaword	.LLASFDE1-.LLframe1	! FDE CIE Pointer
	.uaword	%r_disp32(.LLFB1)	! FDE Initial Location
	.uaword	.LLFE1-.LLFB1	! FDE Address Range
	.uleb128	0x4	! FDE Augmentation Section Length 
	.uaword	0x0
	.byte	0x4	! DW_CFA_advance_loc4
	.uaword	.LLCFI0-.LLFB1
	.byte	0xd	! DW_CFA_def_cfa_register
	.uleb128	0x1e
	.byte	0x2d	! DW_CFA_GNU_window_save
	.byte	0x9	! DW_CFA_register
	.uleb128	0xf
	.uleb128	0x1f
	.align	8	! FDE Padding
.LLEFDE1:
.LLSFDE3:
	.uaword	.LLEFDE3-.LLASFDE3	! FDE Length
.LLASFDE3:
	.uaword	.LLASFDE3-.LLframe1	! FDE CIE Pointer
	.uaword	%r_disp32(.LLFB2)	! FDE Initial Location
	.uaword	.LLFE2-.LLFB2	! FDE Address Range
	.uleb128	0x4	! FDE Augmentation Section Length 
	.uaword	0x0
	.byte	0x4	! DW_CFA_advance_loc4
	.uaword	.LLCFI1-.LLFB2
	.byte	0xd	! DW_CFA_def_cfa_register
	.uleb128	0x1e
	.byte	0x2d	! DW_CFA_GNU_window_save
	.byte	0x9	! DW_CFA_register
	.uleb128	0xf
	.uleb128	0x1f
	.align	8	! FDE Padding
.LLEFDE3:
.LLSFDE7:
	.uaword	.LLEFDE7-.LLASFDE7	! FDE Length
.LLASFDE7:
	.uaword	.LLASFDE7-.LLframe1	! FDE CIE Pointer
	.uaword	%r_disp32(.LLFB4)	! FDE Initial Location
	.uaword	.LLFE4-.LLFB4	! FDE Address Range
	.uleb128	0x4	! FDE Augmentation Section Length 
	.uaword	0x0
	.byte	0x4	! DW_CFA_advance_loc4
	.uaword	.LLCFI2-.LLFB4
	.byte	0xd	! DW_CFA_def_cfa_register
	.uleb128	0x1e
	.byte	0x2d	! DW_CFA_GNU_window_save
	.byte	0x9	! DW_CFA_register
	.uleb128	0xf
	.uleb128	0x1f
	.align	8	! FDE Padding
.LLEFDE7:
.LLSFDE9:
	.uaword	.LLEFDE9-.LLASFDE9	! FDE Length
.LLASFDE9:
	.uaword	.LLASFDE9-.LLframe1	! FDE CIE Pointer
	.uaword	%r_disp32(.LLFB5)	! FDE Initial Location
	.uaword	.LLFE5-.LLFB5	! FDE Address Range
	.uleb128	0x4	! FDE Augmentation Section Length 
	.uaword	0x0
	.byte	0x4	! DW_CFA_advance_loc4
	.uaword	.LLCFI3-.LLFB5
	.byte	0xd	! DW_CFA_def_cfa_register
	.uleb128	0x1e
	.byte	0x2d	! DW_CFA_GNU_window_save
	.byte	0x9	! DW_CFA_register
	.uleb128	0xf
	.uleb128	0x1f
	.align	8	! FDE Padding
.LLEFDE9:
.LLSFDE11:
	.uaword	.LLEFDE11-.LLASFDE11	! FDE Length
.LLASFDE11:
	.uaword	.LLASFDE11-.LLframe1	! FDE CIE Pointer
	.uaword	%r_disp32(.LLFB6)	! FDE Initial Location
	.uaword	.LLFE6-.LLFB6	! FDE Address Range
	.uleb128	0x4	! FDE Augmentation Section Length 
	.uaword	%r_disp32(.LLLSDA6)
	.byte	0x4	! DW_CFA_advance_loc4
	.uaword	.LLCFI4-.LLFB6
	.byte	0xd	! DW_CFA_def_cfa_register
	.uleb128	0x1e
	.byte	0x2d	! DW_CFA_GNU_window_save
	.byte	0x9	! DW_CFA_register
	.uleb128	0xf
	.uleb128	0x1f
	.align	8	! FDE Padding
.LLEFDE11:
.LLSFDE13:
	.uaword	.LLEFDE13-.LLASFDE13	! FDE Length
.LLASFDE13:
	.uaword	.LLASFDE13-.LLframe1	! FDE CIE Pointer
	.uaword	%r_disp32(.LLFB7)	! FDE Initial Location
	.uaword	.LLFE7-.LLFB7	! FDE Address Range
	.uleb128	0x4	! FDE Augmentation Section Length 
	.uaword	%r_disp32(.LLLSDA7)
	.byte	0x4	! DW_CFA_advance_loc4
	.uaword	.LLCFI5-.LLFB7
	.byte	0xd	! DW_CFA_def_cfa_register
	.uleb128	0x1e
	.byte	0x2d	! DW_CFA_GNU_window_save
	.byte	0x9	! DW_CFA_register
	.uleb128	0xf
	.uleb128	0x1f
	.align	8	! FDE Padding
.LLEFDE13:
.LLSFDE15:
	.uaword	.LLEFDE15-.LLASFDE15	! FDE Length
.LLASFDE15:
	.uaword	.LLASFDE15-.LLframe1	! FDE CIE Pointer
	.uaword	%r_disp32(.LLFB8)	! FDE Initial Location
	.uaword	.LLFE8-.LLFB8	! FDE Address Range
	.uleb128	0x4	! FDE Augmentation Section Length 
	.uaword	0x0
	.byte	0x4	! DW_CFA_advance_loc4
	.uaword	.LLCFI6-.LLFB8
	.byte	0xd	! DW_CFA_def_cfa_register
	.uleb128	0x1e
	.byte	0x2d	! DW_CFA_GNU_window_save
	.byte	0x9	! DW_CFA_register
	.uleb128	0xf
	.uleb128	0x1f
	.align	8	! FDE Padding
.LLEFDE15:
.LLSFDE17:
	.uaword	.LLEFDE17-.LLASFDE17	! FDE Length
.LLASFDE17:
	.uaword	.LLASFDE17-.LLframe1	! FDE CIE Pointer
	.uaword	%r_disp32(.LLFB9)	! FDE Initial Location
	.uaword	.LLFE9-.LLFB9	! FDE Address Range
	.uleb128	0x4	! FDE Augmentation Section Length 
	.uaword	0x0
	.byte	0x4	! DW_CFA_advance_loc4
	.uaword	.LLCFI7-.LLFB9
	.byte	0xd	! DW_CFA_def_cfa_register
	.uleb128	0x1e
	.byte	0x2d	! DW_CFA_GNU_window_save
	.byte	0x9	! DW_CFA_register
	.uleb128	0xf
	.uleb128	0x1f
	.align	8	! FDE Padding
.LLEFDE17:
.LLSFDE19:
	.uaword	.LLEFDE19-.LLASFDE19	! FDE Length
.LLASFDE19:
	.uaword	.LLASFDE19-.LLframe1	! FDE CIE Pointer
	.uaword	%r_disp32(.LLFB10)	! FDE Initial Location
	.uaword	.LLFE10-.LLFB10	! FDE Address Range
	.uleb128	0x4	! FDE Augmentation Section Length 
	.uaword	0x0
	.byte	0x4	! DW_CFA_advance_loc4
	.uaword	.LLCFI8-.LLFB10
	.byte	0xd	! DW_CFA_def_cfa_register
	.uleb128	0x1e
	.byte	0x2d	! DW_CFA_GNU_window_save
	.byte	0x9	! DW_CFA_register
	.uleb128	0xf
	.uleb128	0x1f
	.align	8	! FDE Padding
.LLEFDE19:
.LLSFDE21:
	.uaword	.LLEFDE21-.LLASFDE21	! FDE Length
.LLASFDE21:
	.uaword	.LLASFDE21-.LLframe1	! FDE CIE Pointer
	.uaword	%r_disp32(.LLFB11)	! FDE Initial Location
	.uaword	.LLFE11-.LLFB11	! FDE Address Range
	.uleb128	0x4	! FDE Augmentation Section Length 
	.uaword	0x0
	.byte	0x4	! DW_CFA_advance_loc4
	.uaword	.LLCFI9-.LLFB11
	.byte	0xd	! DW_CFA_def_cfa_register
	.uleb128	0x1e
	.byte	0x2d	! DW_CFA_GNU_window_save
	.byte	0x9	! DW_CFA_register
	.uleb128	0xf
	.uleb128	0x1f
	.align	8	! FDE Padding
.LLEFDE21:
	.weak	_ZZNKSt12_String_baseIcSt20__iostring_allocatorIcEE21_M_throw_length_errorEvEs
	.section	".rodata._ZZNKSt12_String_baseIcSt20__iostring_allocatorIcEE21_M_throw_length_errorEvEs",#alloc
	.align	8
_ZZNKSt12_String_baseIcSt20__iostring_allocatorIcEE21_M_throw_length_errorEvEs:
	.byte	98
	.byte	97
	.byte	115
	.byte	105
	.byte	99
	.byte	95
	.byte	115
	.byte	116
	.byte	114
	.byte	105
	.byte	110
	.byte	103
	.skip	1
	.type	_ZZNKSt12_String_baseIcSt20__iostring_allocatorIcEE21_M_throw_length_errorEvEs,#object
	.size	_ZZNKSt12_String_baseIcSt20__iostring_allocatorIcEE21_M_throw_length_errorEvEs,.-_ZZNKSt12_String_baseIcSt20__iostring_allocatorIcEE21_M_throw_length_errorEvEs
	.section	".rodata"
	.align	8
.LR0..0.8:
	.byte	118
	.byte	111
	.byte	105
	.byte	100
	.byte	32
	.byte	80
	.byte	97
	.byte	114
	.byte	116
	.byte	105
	.byte	99
	.byte	108
	.byte	101
	.byte	58
	.byte	58
	.byte	114
	.byte	101
	.byte	115
	.byte	116
	.byte	111
	.byte	114
	.byte	101
	.byte	40
	.byte	95
	.byte	73
	.byte	79
	.byte	95
	.byte	70
	.byte	73
	.byte	76
	.byte	69
	.byte	32
	.byte	42
	.byte	41
	.skip	1
	.type	.LR0..0.8,#object
	.size	.LR0..0.8,.-.LR0..0.8
	.section	".rodata"
	.align	8
.LR0..1.7:
	.byte	71
	.byte	114
	.byte	97
	.byte	118
	.byte	105
	.byte	116
	.byte	121
	.byte	58
	.byte	58
	.byte	71
	.byte	114
	.byte	97
	.byte	118
	.byte	105
	.byte	116
	.byte	121
	.byte	40
	.byte	105
	.byte	110
	.byte	116
	.byte	41
	.skip	1
	.type	.LR0..1.7,#object
	.size	.LR0..1.7,.-.LR0..1.7
	.section	".data"
	.align	16
_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf:
	.skip	1179648
	.type	_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf,#object
	.size	_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf,.-_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf
	.section	".rodata"
	.align	8
.LR0..2.5:
	.byte	100
	.byte	111
	.byte	117
	.byte	98
	.byte	108
	.byte	101
	.byte	32
	.byte	112
	.byte	111
	.byte	119
	.byte	95
	.byte	111
	.byte	110
	.byte	101
	.byte	95
	.byte	110
	.byte	116
	.byte	104
	.byte	95
	.byte	113
	.byte	117
	.byte	97
	.byte	110
	.byte	116
	.byte	40
	.byte	100
	.byte	111
	.byte	117
	.byte	98
	.byte	108
	.byte	101
	.byte	41
	.byte	32
	.byte	91
	.byte	119
	.byte	105
	.byte	116
	.byte	104
	.byte	32
	.byte	78
	.byte	32
	.byte	61
	.byte	32
	.byte	54
	.byte	93
	.skip	1
	.type	.LR0..2.5,#object
	.size	.LR0..2.5,.-.LR0..2.5
	.section	".rodata"
	.align	8
.LR0..3.4:
	.byte	84
	.byte	32
	.byte	42
	.byte	97
	.byte	108
	.byte	108
	.byte	111
	.byte	99
	.byte	97
	.byte	116
	.byte	101
	.byte	40
	.byte	117
	.byte	110
	.byte	115
	.byte	105
	.byte	103
	.byte	110
	.byte	101
	.byte	100
	.byte	32
	.byte	108
	.byte	111
	.byte	110
	.byte	103
	.byte	41
	.byte	32
	.byte	91
	.byte	119
	.byte	105
	.byte	116
	.byte	104
	.byte	32
	.byte	84
	.byte	32
	.byte	61
	.byte	32
	.byte	71
	.byte	114
	.byte	97
	.byte	118
	.byte	105
	.byte	116
	.byte	121
	.byte	58
	.byte	58
	.byte	71
	.byte	80
	.byte	97
	.byte	114
	.byte	116
	.byte	105
	.byte	99
	.byte	108
	.byte	101
	.byte	44
	.byte	32
	.byte	97
	.byte	108
	.byte	105
	.byte	103
	.byte	110
	.byte	32
	.byte	61
	.byte	32
	.byte	49
	.byte	50
	.byte	56
	.byte	85
	.byte	76
	.byte	93
	.skip	1
	.type	.LR0..3.4,#object
	.size	.LR0..3.4,.-.LR0..3.4
	.section	".rodata"
	.align	8
.LR0..4.3:
	.byte	84
	.byte	32
	.byte	42
	.byte	97
	.byte	108
	.byte	108
	.byte	111
	.byte	99
	.byte	97
	.byte	116
	.byte	101
	.byte	40
	.byte	117
	.byte	110
	.byte	115
	.byte	105
	.byte	103
	.byte	110
	.byte	101
	.byte	100
	.byte	32
	.byte	108
	.byte	111
	.byte	110
	.byte	103
	.byte	41
	.byte	32
	.byte	91
	.byte	119
	.byte	105
	.byte	116
	.byte	104
	.byte	32
	.byte	84
	.byte	32
	.byte	61
	.byte	32
	.byte	71
	.byte	114
	.byte	97
	.byte	118
	.byte	105
	.byte	116
	.byte	121
	.byte	58
	.byte	58
	.byte	71
	.byte	80
	.byte	114
	.byte	101
	.byte	100
	.byte	105
	.byte	99
	.byte	116
	.byte	111
	.byte	114
	.byte	44
	.byte	32
	.byte	97
	.byte	108
	.byte	105
	.byte	103
	.byte	110
	.byte	32
	.byte	61
	.byte	32
	.byte	49
	.byte	50
	.byte	56
	.byte	85
	.byte	76
	.byte	93
	.skip	1
	.type	.LR0..4.3,#object
	.size	.LR0..4.3,.-.LR0..4.3
	.section	".rodata"
	.align	8
.LR0..5.2:
	.byte	84
	.byte	32
	.byte	42
	.byte	97
	.byte	108
	.byte	108
	.byte	111
	.byte	99
	.byte	97
	.byte	116
	.byte	101
	.byte	40
	.byte	117
	.byte	110
	.byte	115
	.byte	105
	.byte	103
	.byte	110
	.byte	101
	.byte	100
	.byte	32
	.byte	108
	.byte	111
	.byte	110
	.byte	103
	.byte	41
	.byte	32
	.byte	91
	.byte	119
	.byte	105
	.byte	116
	.byte	104
	.byte	32
	.byte	84
	.byte	32
	.byte	61
	.byte	32
	.byte	118
	.byte	52
	.byte	114
	.byte	56
	.byte	44
	.byte	32
	.byte	97
	.byte	108
	.byte	105
	.byte	103
	.byte	110
	.byte	32
	.byte	61
	.byte	32
	.byte	49
	.byte	50
	.byte	56
	.byte	85
	.byte	76
	.byte	93
	.skip	1
	.type	.LR0..5.2,#object
	.size	.LR0..5.2,.-.LR0..5.2
	.section	".bss"
	.align	8
.LB0..119.1:
	.skip	8
	.type	.LB0..119.1,#object
	.size	.LB0..119.1,.-.LB0..119.1
	.section	".rodata"
	.align	8
.LR0.cnt.7:
	.word	0XC0080000,0
	.type	.LR0.cnt.7,#object
	.size	.LR0.cnt.7,.-.LR0.cnt.7
	.section	".rodata"
	.align	8
.LR0.cnt.6:
	.word	0X3FD00000,0
	.type	.LR0.cnt.6,#object
	.size	.LR0.cnt.6,.-.LR0.cnt.6
	.section	".rodata"
	.align	8
.LR0.cnt.5:
	.word	0X3FE00000,0
	.type	.LR0.cnt.5,#object
	.size	.LR0.cnt.5,.-.LR0.cnt.5
	.section	".rodata"
	.align	8
.LR0.cnt.4:
	.word	0X3FF00000,0
	.type	.LR0.cnt.4,#object
	.size	.LR0.cnt.4,.-.LR0.cnt.4
	.section	".rodata"
	.align	8
.LR0.cnt.3:
	.word	0,0
	.type	.LR0.cnt.3,#object
	.size	.LR0.cnt.3,.-.LR0.cnt.3
	.section	".rodata"
	.align	8
.LR0.cnt.2:
	.word	0X3FC99999,0X9999999A
	.type	.LR0.cnt.2,#object
	.size	.LR0.cnt.2,.-.LR0.cnt.2
	.section	".rodata"
	.align	8
.LR0.cnt.1:
	.word	0X3FD55555,0X55555555
	.type	.LR0.cnt.1,#object
	.size	.LR0.cnt.1,.-.LR0.cnt.1
	.section	".data"
	.align	16
.LS0:
	.align	8
.LS0.cnt.8:
	.word	1065353216
	.type	.LS0.cnt.8,#object
	.size	.LS0.cnt.8,.-.LS0.cnt.8
