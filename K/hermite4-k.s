	.ident	"$Options: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) --preinclude //opt/FJSVfxlang/1.2.1/bin/../lib/FCC.pre --g++ -D__FUJITSU -Dunix -Dsparc -D__sparc__ -D__unix -D__sparc -D__BUILTIN_VA_ARG_INCR -D_OPENMP=200805 -D__PRAGMA_REDEFINE_EXTNAME -D__FCC_VERSION=600 -D__USER_LABEL_PREFIX__= -D__OPTIMIZE__ -D__HPC_ACE__ -D__ELF__ -D__linux -Asystem(unix) -Dlinux -D__LIBC_6B -D_LP64 -D__LP64__ --K=omp -DFOURTH -DHPC_ACE_GRAVITY -I/opt/FJSVfxlang/1.2.1/include/mpi/fujitsu --K=noocl -D_REENTRANT -D__MT__ --lp --zmode=64 --sys_include=//opt/FJSVfxlang/1.2.1/bin/../include/c++/std --sys_include=//opt/FJSVfxlang/1.2.1/bin/../include/c++ --sys_include=//opt/FJSVfxlang/1.2.1/bin/../include --sys_include=/opt/FJSVXosDevkit/sparc64fx/target/usr/include --K=opt -D__sparcv9 -D__sparc_v9__ -D__arch64__ --exceptions ../SRC/hermite4-k.cpp -- -ncmdname=FCCpx -Nnoline -Kdalign -zobe=no-static-clump -zobe=cplus -O3 -x- -KSPARC64IXfx,dalign,ns,mfunc,lib,eval,rdconv,prefetch_conditional,fp_contract,fp_relaxed,ilfunc,fast_matmul -Kthreadsafe -O3 -x- -KSPARC64IXfx,dalign,ns,mfunc,lib,eval,rdconv,prefetch_conditional,fp_contract,fp_relaxed,ilfunc,fast_matmul,uxsimd,optmsg=2 -x32 -Nsrc -Kopenmp,threadsafe -KLP -zsrc=../SRC/hermite4-k.cpp hermite4-k.s $"
	.file	"hermite4-k.cpp"
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite4-k.cpp _ZNKSt12_String_baseIcSt20__iostring_allocatorIcEE21_M_throw_length_errorEv $"
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
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite4-k.cpp _ZNSt12_String_baseIcSt20__iostring_allocatorIcEE17_M_allocate_blockEm $"
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


.L1143:

/*    637 */	ret
	restore



.L210:


/*     61 */	cmp	%i1,257

/*     61 */	bleu,pt	%xcc, .L1140
/*     61 */	add	%i0,16,%o0


.L1139:

/*    458 */	cmp	%i1,-1

/*    458 */	bgu,pn	%xcc, .L1112
	nop


.L1120:


/*    123 */	call	_Znwm
/*    123 */	mov	%i1,%o0


.L4570:

/*    123 */	cmp	%o0,%g0

/*    123 */	be,pt	%xcc, .L1123
	nop


.L1127:


.L1140:

/*    660 */	add	%o0,%i1,%i1

/*    657 */	stx	%o0,[%i0]

/*    658 */	stx	%o0,[%i0+8]

/*    660 */	stx	%i1,[%i0+280]

/*      0 */	ret
	restore



.L1123:


/*    123 */	call	__cxa_allocate_exception
/*    123 */	mov	8,%o0
/*    123 */	mov	%o0,%l1
/*    123 */	call	_ZNSt9bad_allocC1Ev
/*    123 */	mov	%l1,%o0


.L5009:

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


.L1112:


/*    459 */	call	__cxa_allocate_exception
/*    459 */	mov	8,%o0
/*    459 */	mov	%o0,%l0
/*    459 */	call	_ZNSt9bad_allocC1Ev
/*    459 */	mov	%l0,%o0


.L5008:

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
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite4-k.cpp _ZN7Gravity6GForceC1Ev $"
	.section	".text._ZN7Gravity6GForceC1Ev",#alloc,#execinstr

	.weak	_ZN7Gravity6GForceC1Ev
	.align	64
_ZN7Gravity6GForceC1Ev:
.LLFB3:
.L563:

/*     31 */

.L564:


/*     25 */	sxar2
/*     25 */	fzero,s	%f32
/*     25 */	std,s	%f32,[%o0]


/*     25 */	sxar2
/*     25 */	std,s	%f32,[%o0+16]
/*     25 */	std,s	%f32,[%o0+32]


/*     25 */	sxar2
/*     25 */	std,s	%f32,[%o0+48]
/*     25 */	std,s	%f32,[%o0+64]

/*     25 */	sxar1
/*     25 */	std,s	%f32,[%o0+80]

/*     25 */	retl
	nop



.L565:


.LLFE3:
	.size	_ZN7Gravity6GForceC1Ev,.-_ZN7Gravity6GForceC1Ev
	.type	_ZN7Gravity6GForceC1Ev,#function
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite4-k.cpp _ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE $"
	.section	".text"
	.global	_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE
	.align	64
_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE:
.LLFB4:
.L575:

/*      8 */	save	%sp,-272,%sp
.LLCFI2:
/*      8 */	stx	%i2,[%fp+2191]
/*      8 */	stx	%i3,[%fp+2199]

.L576:

/*     26 */	sxar1
/*     26 */	fmovd	%f2,%f258


/*     15 */	srl	%i0,31,%g1

/*     15 */	add	%g1,%i0,%g1

/*     15 */	sra	%g1,1,%g1

/*     15 */	stw	%g1,[%fp+2031]

/*     26 */	sxar1
/*     26 */	std,s	%f2,[%fp+1967]

/*     16 *//*     16 */	sethi	%h44(_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE._OMP_1),%o0
/*     16 */	mov	%fp,%o1
/*     16 */	or	%o0,%m44(_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE._OMP_1),%o0
/*     16 */	mov	%g0,%o2
/*     16 */	sllx	%o0,12,%o0
/*     16 */	call	__mpc_opar
/*     16 */	or	%o0,%l44(_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE._OMP_1),%o0

/*     35 */
/*     35 */	ret
	restore



.L602:


.LLFE4:
	.size	_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE,.-_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE
	.type	_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE,#function
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite4-k.cpp _ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE._OMP_1 $"
	.section	".text"
	.align	64
_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE._OMP_1:
.LLFB5:
.L4754:

/*     16 */	save	%sp,-864,%sp
.LLCFI3:
/*     16 */	stx	%i0,[%fp+2175]
/*     16 */	stx	%i3,[%fp+2199]
/*     16 */	stx	%i0,[%fp+2175]

.L4755:

/*     16 *//*     16 */	ldsw	[%i0+2035],%g3
/*     16 */
/*     16 */
/*     16 */
/*     17 */	sxar2
/*     17 */	ldsw	[%i0+2031],%xg16
/*     17 */	cmp	%xg16,%g0
/*     17 */	ble	.L4770
/*     17 */	mov	%g0,%o0


.L4756:


/*     17 */	sxar2
/*     17 */	sethi	%h44(.LR0.cnt.4),%xg4
/*     17 */	sethi	%h44(.LR0.cnt.1),%xg5


/*     17 */	sxar2
/*     17 */	or	%xg4,%m44(.LR0.cnt.4),%xg4
/*     17 */	or	%xg5,%m44(.LR0.cnt.1),%xg5


/*     17 */	sxar2
/*     17 */	sllx	%xg4,12,%xg4
/*     17 */	sra	%xg16,%g0,%g2


/*     17 */	sxar2
/*     17 */	or	%xg4,%l44(.LR0.cnt.4),%xg4
/*     17 */	sllx	%xg5,12,%xg5

/*     17 */	mov	1,%g1

/*    ??? */	stx	%g2,[%fp+1431]


/*     17 */	sxar2
/*     17 */	or	%xg5,%l44(.LR0.cnt.1),%xg5
/*     17 */	ldd	[%xg4],%f52



/*     17 */	sxar2
/*     17 */	ldd	[%xg4],%f308
/*     17 */	ldd	[%xg5],%f54

/*     17 */	sxar1
/*     17 */	ldd	[%xg5],%f310


/*     17 */	stx	%g1,[%fp+2031]


/*     17 */	sxar2
/*    ??? */	std,s	%f52,[%fp+1711]
/*    ??? */	std,s	%f54,[%fp+1695]

.L4757:


/*     17 */	sxar2
/*     17 */	add	%fp,2039,%xg27
/*     17 */	mov	1,%xg28


/*     17 */	sxar2
/*    ??? */	stw	%xg28,[%fp+1451]
/*     17 */	add	%fp,2023,%xg29


/*     17 */	sxar2
/*     17 */	add	%fp,2031,%xg30
/*     17 */	sra	%xg28,%g0,%xg31


/*     17 */	sxar2
/*    ??? */	stx	%xg27,[%fp+1423]
/*    ??? */	stx	%xg29,[%fp+1415]


/*     17 */	sxar2
/*    ??? */	stx	%xg30,[%fp+1439]
/*    ??? */	stx	%xg31,[%fp+1455]

.L4758:

/*     17 */	sra	%o0,%g0,%o0

/*     17 */	stx	%g0,[%sp+2223]

/*     17 */	mov	1,%o2

/*     17 */	mov	%g0,%o3

/*     17 */	sxar1
/*    ??? */	ldx	[%fp+1439],%xg24

/*    ??? */	ldx	[%fp+1431],%o1

/*    ??? */	ldx	[%fp+1423],%o4


/*     17 */	stx	%g0,[%sp+2231]


/*     17 */	sxar2
/*     17 */	stx	%xg24,[%sp+2239]
/*     17 */	ldx	[%fp+2199],%xg25

/*     17 */	sxar1
/*     17 */	stx	%xg25,[%sp+2247]

/*     17 */	call	__mpc_ostd_th
/*    ??? */	ldx	[%fp+1415],%o5
/*     17 */	sxar2
/*     17 */	ldx	[%fp+2031],%xg26
/*     17 */	cmp	%xg26,%g0
/*     17 */	ble,pn	%xcc, .L4770
	nop


.L4759:

/*     17 */	ldx	[%fp+2039],%o0

/*     17 */	ldx	[%fp+2023],%g2


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1455],%xg22
/*    ??? */	ldsw	[%fp+1451],%xg23

/*     17 */	sxar1
/*     17 */	ldd,s	[%i0+1967],%f34

/*     17 */	sra	%o0,%g0,%o0

/*     17 */	sra	%g2,%g0,%g2

/*     17 */	sub	%g2,%o0,%g2

/*     17 */	sxar1
/*     17 */	sra	%o0,%g0,%xg6

/*     17 */	add	%g2,1,%g2

/*     17 */	sxar1
/*     17 */	sllx	%xg6,3,%g3

/*     17 */	sra	%g2,%g0,%g2


/*     17 */	sxar2
/*     17 */	sub	%g3,%xg6,%g3
/*     17 */	sub	%xg22,%g2,%g2

/*     17 */	sllx	%g3,5,%g5



/*     17 */	sxar2
/*     17 */	srax	%g2,32,%xg7
/*     17 */	and	%g2,%xg7,%g2

/*     17 */	sxar1
/*     17 */	sub	%xg23,%g2,%g2

/*     17 */	subcc	%g2,2,%g2

/*     17 */	bneg	.L4773
/*     17 */	sllx	%g3,4,%g3


.L4776:

/*     17 */	ldx	[%i0+2191],%g4

/*     17 */	sxar1
/*     17 */	ldx	[%i0+2199],%xg8

/*     17 */	cmp	%g2,18


/*     17 */	sxar2
/*     17 */	add	%g4,112,%xg9
/*     17 */	add	%g4,64,%xg10


/*     17 */	sxar2
/*     17 */	add	%g4,128,%xg11
/*     17 */	add	%g4,176,%xg12


/*     17 */	sxar2
/*    ??? */	stx	%xg8,[%fp+1655]
/*     17 */	add	%xg8,64,%xg13


/*     17 */	sxar2
/*     17 */	add	%g4,80,%xg14
/*    ??? */	stx	%xg9,[%fp+1687]


/*     17 */	sxar2
/*     17 */	add	%g4,144,%xg15
/*     17 */	add	%g4,192,%xg16


/*     17 */	sxar2
/*    ??? */	stx	%xg10,[%fp+1671]
/*     17 */	add	%xg8,32,%xg17


/*     17 */	sxar2
/*     17 */	add	%g4,96,%xg18
/*    ??? */	stx	%xg11,[%fp+1679]


/*     17 */	sxar2
/*     17 */	add	%g4,160,%xg19
/*     17 */	add	%g4,208,%xg20


/*     17 */	sxar2
/*    ??? */	stx	%xg12,[%fp+1663]
/*     17 */	add	%xg8,16,%o7


/*     17 */	sxar2
/*     17 */	add	%xg8,48,%xg21
/*    ??? */	stx	%xg13,[%fp+1647]

/*     17 */	add	%g4,16,%o2


/*     17 */	sxar2
/*     17 */	add	%xg8,80,%o5
/*    ??? */	stx	%xg14,[%fp+1631]

/*     17 */	add	%g4,32,%o4


/*     17 */	sxar2
/*     17 */	add	%xg8,96,%o3
/*    ??? */	stx	%xg15,[%fp+1623]



/*     17 */	sxar2
/*    ??? */	stx	%xg16,[%fp+1639]
/*    ??? */	stx	%xg17,[%fp+1591]


/*     17 */	sxar2
/*    ??? */	stx	%xg18,[%fp+1599]
/*    ??? */	stx	%xg19,[%fp+1615]


/*     17 */	sxar2
/*    ??? */	stx	%xg20,[%fp+1607]
/*    ??? */	stx	%xg21,[%fp+1583]

/*     17 */	bl	.L4959
/*     17 */	add	%g4,48,%o1


.L4955:


.L4964:


/*     17 */	sxar2
/*     17 */	add	%g5,224,%xg8
/*    ??? */	ldx	[%fp+1663],%xg18


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1679],%xg21
/*     17 */	add	%g5,448,%xg19


/*     17 */	sxar2
/*     17 */	add	%g5,672,%xg3
/*    ??? */	ldx	[%fp+1607],%xg22


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1623],%xg23
/*     17 */	add	%o2,%xg8,%xg30


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1599],%xg27
/*    ??? */	ldx	[%fp+1687],%xg28


/*     17 */	sxar2
/*     17 */	add	%g4,%g5,%xg25
/*     17 */	add	%g4,%xg8,%xg4


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1687],%xg17
/*    ??? */	ldx	[%fp+1631],%xg29


/*     17 */	sxar2
/*     17 */	add	%g5,1344,%xg2
/*     17 */	add	%xg18,%g5,%xg11

/*     17 */	sxar1
/*     17 */	add	%xg21,%g5,%xg14

/*    ??? */	ldx	[%fp+1679],%g1


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1663],%xg0
/*    ??? */	ldx	[%fp+1615],%xg20


/*     17 */	sxar2
/*     17 */	add	%xg22,%g5,%xg15
/*     17 */	add	%xg23,%xg8,%xg16


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1639],%xg24
/*     17 */	add	%xg23,%g5,%xg18


/*     17 */	sxar2
/*     17 */	add	%xg27,%g5,%xg21
/*     17 */	add	%xg17,%g5,%xg9


/*     17 */	sxar2
/*     17 */	add	%xg28,%xg19,%xg22
/*    ??? */	ldx	[%fp+1671],%xg31


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg9],%f32
/*     17 */	add	%xg29,%xg8,%xg23


/*     17 */	sxar2
/*     17 */	add	%g1,%xg8,%xg27
/*     17 */	add	%xg17,%xg8,%xg10


/*     17 */	sxar2
/*     17 */	add	%xg0,%xg8,%xg28
/*     17 */	ldd,s	[%xg23],%f78


/*     17 */	sxar2
/*     17 */	add	%xg17,%xg3,%xg12
/*     17 */	add	%xg20,%g5,%xg13


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg22],%f48
/*     17 */	ldd,s	[%xg28],%f88


/*     17 */	sxar2
/*     17 */	add	%xg24,%xg8,%xg17
/*     17 */	add	%xg24,%g5,%xg20


/*     17 */	sxar2
/*    ??? */	ldd,s	[%fp+1695],%f86
/*     17 */	ldd,s	[%xg27],%f84


/*     17 */	sxar2
/*     17 */	fsubd,s	%f34,%f32,%f32
/*     17 */	add	%xg29,%g5,%xg24


/*     17 */	sxar2
/*     17 */	add	%xg31,%g5,%xg26
/*    ??? */	ldd,s	[%fp+1711],%f246


/*     17 */	sxar2
/*     17 */	add	%o4,%g5,%xg29
/*     17 */	add	%xg0,%xg19,%xg31


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1687],%xg5
/*    ??? */	ldx	[%fp+1615],%xg6

/*     17 */	add	%g5,1120,%g1


/*     17 */	sxar2
/*     17 */	fsubd,s	%f34,%f48,%f48
/*    ??? */	ldx	[%fp+1679],%xg7


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1631],%xg27
/*     17 */	ldd,s	[%xg10],%f36


/*     17 */	sxar2
/*     17 */	add	%g5,896,%xg0
/*     17 */	add	%o2,%g5,%xg9

/*     17 */	sxar1
/*    ??? */	ldx	[%fp+1591],%xg28

/*    ??? */	stw	%g2,[%fp+1575]


/*     17 */	sxar2
/*     17 */	add	%xg5,%g1,%xg1
/*     17 */	add	%g4,%xg19,%xg5

/*    ??? */	stw	%o0,[%fp+1579]


/*     17 */	sxar2
/*     17 */	fmuld,s	%f86,%f32,%f38
/*     17 */	fmuld,s	%f246,%f32,%f46


/*     17 */	sxar2
/*     17 */	add	%xg6,%xg19,%xg6
/*    ??? */	stx	%xg27,[%fp+1463]


/*     17 */	sxar2
/*     17 */	fsubd,s	%f34,%f36,%f36
/*     17 */	add	%xg7,%xg19,%xg7


/*     17 */	sxar2
/*    ??? */	stx	%xg28,[%fp+1471]
/*     17 */	ldd,s	[%xg29],%f90


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg30],%f92
/*     17 */	ldd,s	[%xg31],%f116


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg4],%f108
/*     17 */	ldd,s	[%xg11],%f54


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg13],%f56
/*     17 */	ldd,s	[%xg15],%f60


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg14],%f52
/*    ??? */	ldx	[%fp+1687],%xg29


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1583],%xg30
/*     17 */	fmuld,s	%f86,%f36,%f42


/*     17 */	sxar2
/*     17 */	fmuld,s	%f246,%f36,%f44
/*    ??? */	ldx	[%fp+1647],%xg31

/*    ??? */	ldx	[%fp+1655],%g2

/*    ??? */	ldx	[%fp+1599],%o0


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1615],%xg4
/*     17 */	ldd,s	[%xg17],%f66


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f38,%f54,%f52,%f50
/*     17 */	fmaddd,s	%f38,%f60,%f56,%f58


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg16],%f62
/*     17 */	ldd,s	[%xg12],%f40


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f46,%f54,%f52,%f54
/*     17 */	ldd,s	[%xg20],%f70


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg18],%f68
/*    ??? */	stx	%xg29,[%fp+1479]


/*     17 */	sxar2
/*    ??? */	stx	%xg30,[%fp+1487]
/*     17 */	fsubd,s	%f34,%f40,%f40


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f42,%f66,%f62,%f64
/*    ??? */	stx	%xg31,[%fp+1495]


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f38,%f70,%f68,%f38
/*     17 */	fmaddd,s	%f46,%f70,%f68,%f70

/*    ??? */	stx	%g2,[%fp+1503]

/*    ??? */	stx	%o0,[%fp+1511]


/*     17 */	sxar2
/*    ??? */	stx	%xg4,[%fp+1519]
/*     17 */	ldd,s	[%xg21],%f76


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg24],%f80
/*     17 */	ldd,s	[%xg26],%f74


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1639],%xg10
/*     17 */	fmuld,s	%f86,%f40,%f72


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f44,%f64,%f78,%f64
/*    ??? */	ldx	[%fp+1623],%xg11


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1607],%xg12
/*     17 */	ldd,s	[%xg25],%f82


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1671],%xg13
/*     17 */	fmaddd,s	%f46,%f58,%f76,%f58


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f46,%f38,%f80,%f38
/*    ??? */	ldx	[%fp+1663],%xg14


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f46,%f50,%f74,%f50
/*    ??? */	ldx	[%fp+1679],%xg15


/*     17 */	sxar2
/*    ??? */	stx	%xg10,[%fp+1527]
/*    ??? */	stx	%xg11,[%fp+1535]


/*     17 */	sxar2
/*    ??? */	stx	%xg12,[%fp+1543]
/*    ??? */	stx	%xg13,[%fp+1551]


/*     17 */	sxar2
/*    ??? */	stx	%xg14,[%fp+1559]
/*    ??? */	stx	%xg15,[%fp+1567]

.L4760:


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg9],%f242
/* #00002 */	ldx	[%fp+1567],%xg24


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f46,%f60,%f56,%f46
/* #00002 */	ldx	[%fp+1559],%xg25


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1551],%xg26
/* #00002 */	ldx	[%fp+1543],%xg27


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1535],%xg28
/* #00002 */	ldx	[%fp+1527],%xg29


/*     17 */	sxar2
/*     17 */	add	%xg24,%xg3,%xg4
/* #00002 */	ldx	[%fp+1519],%xg30

/*     17 */	sxar1
/*     17 */	add	%xg25,%xg3,%xg9

/* #00002 */	ldx	[%fp+1511],%g2

/*     17 */	sxar1
/*     17 */	ldd,s	[%xg9],%f136

/* #00002 */	ldx	[%fp+1503],%o0


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f32,%f50,%f82,%f50
/* #00002 */	ldx	[%fp+1471],%xg9


/*     17 */	sxar2
/*     17 */	add	%xg26,%xg8,%xg10
/*     17 */	ldd,s	[%xg4],%f132


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1487],%xg4
/*     17 */	fmaddd,s	%f32,%f54,%f74,%f54


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg10],%f56
/* #00002 */	ldx	[%fp+1463],%xg10


/*     17 */	sxar2
/*     17 */	add	%xg2,224,%xg11
/*     17 */	add	%xg27,%xg19,%xg12


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg12],%f114
/*     17 */	ldd,s	[%xg6],%f110


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1479],%xg6
/*     17 */	add	%xg28,%xg3,%xg13


/*     17 */	sxar2
/*     17 */	add	%xg29,%xg3,%xg14
/*     17 */	ldd,s	[%xg14],%f252


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1527],%xg14
/*     17 */	fmaddd,s	%f42,%f88,%f84,%f240


/*     17 */	sxar2
/*     17 */	add	%xg2,448,%xg15
/*     17 */	ldd,s	[%xg13],%f248


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f32,%f38,%f242,%f38
/*     17 */	ldd,s	[%xg1],%f94


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1495],%xg1
/*     17 */	add	%xg30,%xg8,%xg16


/*     17 */	sxar2
/*     17 */	add	%xg27,%xg8,%xg17
/*     17 */	ldd,s	[%xg17],%f52


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1551],%xg17
/*     17 */	add	%g2,%xg19,%xg18


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg16],%f254
/*     17 */	add	%o0,%g3,%xg20


/*     17 */	sxar2
/*     17 */	std,s	%f50,[%xg20]
/*     17 */	fmaddd,s	%f32,%f70,%f80,%f70


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1479],%xg20
/*     17 */	add	%xg1,%g3,%xg21


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f32,%f58,%f90,%f58
/*     17 */	std,s	%f54,[%xg21]


/*     17 */	sxar2
/*     17 */	add	%xg4,%g3,%xg22
/* #00002 */	ldx	[%fp+1503],%xg21


/*     17 */	sxar2
/*     17 */	fmuld,s	%f246,%f40,%f244
/*     17 */	add	%xg6,%xg0,%xg23


/*     17 */	sxar2
/*     17 */	add	%xg9,%g3,%xg24
/*     17 */	ldd,s	[%xg18],%f126


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1559],%xg18
/*     17 */	add	%xg10,%xg3,%xg25


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f72,%f252,%f248,%f250
/*     17 */	add	%o7,%g3,%xg26


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f32,%f46,%f76,%f32
/*     17 */	add	%xg28,%xg19,%xg27


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg25],%f96
/* #00002 */	ldx	[%fp+1471],%xg25


/*     17 */	sxar2
/*     17 */	std,s	%f38,[%xg26]
/*     17 */	fmaddd,s	%f36,%f64,%f92,%f64


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1487],%xg26
/*     17 */	add	%g2,%xg8,%xg28


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f42,%f52,%f254,%f42
/*     17 */	ldd,s	[%xg5],%f130


/*     17 */	sxar2
/*     17 */	add	%o5,%g3,%xg29
/*     17 */	fmaddd,s	%f44,%f240,%f56,%f240


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg28],%f98
/* #00002 */	ldx	[%fp+1559],%xg28


/*     17 */	sxar2
/*     17 */	std,s	%f70,[%xg29]
/*     17 */	fmuld,s	%f86,%f48,%f60


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1551],%xg29
/*     17 */	add	%xg14,%xg19,%xg30


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f44,%f88,%f84,%f88
/*     17 */	std,s	%f58,[%xg24]


/*     17 */	sxar2
/*     17 */	fsubd,s	%f34,%f94,%f94
/* #00002 */	ldx	[%fp+1567],%xg24

/*     17 */	sxar1
/*     17 */	add	%o4,%xg19,%xg31

/*     17 */	add	%o3,%g3,%g2


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg31],%f140
/* #00002 */	ldx	[%fp+1535],%xg31


/*     17 */	sxar2
/*     17 */	add	%o2,%xg3,%o0
/*     17 */	fmaddd,s	%f244,%f250,%f96,%f250


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg7],%f100
/*     17 */	ldd,s	[%xg30],%f120

/*     17 */	sxar1
/* #00002 */	ldx	[%fp+1543],%xg30

/*     17 */	add	%o1,%g5,%g5

/*     17 */	sxar1
/*     17 */	ldd,s	[%o0],%f150

/* #00002 */	ldx	[%fp+1519],%o0


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f44,%f66,%f62,%f66
/*     17 */	ldd,s	[%xg27],%f118


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1567],%xg27
/*     17 */	ldd,s	[%g5],%f90


/*     17 */	sxar2
/*     17 */	add	%o4,%xg8,%xg1
/*     17 */	fmaddd,s	%f44,%f42,%f98,%f42


/*     17 */	sxar2
/*     17 */	add	%xg10,%xg19,%xg4
/*     17 */	fmaddd,s	%f36,%f240,%f108,%f240


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg1],%f108
/* #00002 */	ldx	[%fp+1543],%xg1


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f44,%f52,%f254,%f44
/*     17 */	std,s	%f32,[%g2]

/*     17 */	sxar1
/*     17 */	add	%xg17,%xg19,%xg5

/* #00002 */	ldx	[%fp+1527],%g2


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f36,%f88,%f56,%f88
/*     17 */	add	%xg18,%xg0,%xg6


/*     17 */	sxar2
/*     17 */	add	%xg20,%xg11,%xg7
/*     17 */	fmaddd,s	%f60,%f116,%f100,%f102


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg4],%f128
/* #00002 */	ldx	[%fp+1511],%xg4


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg6],%f172
/* #00002 */	ldx	[%fp+1495],%xg6


/*     17 */	sxar2
/*     17 */	add	%g4,%xg3,%xg9
/*     17 */	fmuld,s	%f246,%f48,%f104


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg23],%f122
/* #00002 */	ldx	[%fp+1519],%xg23


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg9],%f166
/* #00002 */	ldx	[%fp+1471],%xg9


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f36,%f66,%f78,%f66
/*     17 */	add	%g3,112,%xg10


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg5],%f124
/* #00002 */	ldx	[%fp+1503],%xg5


/*     17 */	sxar2
/*     17 */	std,s	%f90,[%xg22]
/*     17 */	fmuld,s	%f86,%f94,%f106


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1495],%xg22
/*     17 */	add	%xg21,%xg10,%xg12


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f36,%f42,%f108,%f42
/*     17 */	std,s	%f240,[%xg12]


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f60,%f114,%f110,%f112
/* #00002 */	ldx	[%fp+1535],%xg12


/*     17 */	sxar2
/*     17 */	add	%xg22,%xg10,%xg13
/*     17 */	fmaddd,s	%f36,%f44,%f98,%f36


/*     17 */	sxar2
/*     17 */	std,s	%f88,[%xg13]
/*     17 */	fmaddd,s	%f60,%f120,%f118,%f60


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1511],%xg13
/*     17 */	add	%g4,%xg0,%xg14


/*     17 */	sxar2
/*     17 */	add	%o1,%xg8,%xg8
/*     17 */	add	%xg23,%xg0,%xg16


/*     17 */	sxar2
/*     17 */	fsubd,s	%f34,%f122,%f122
/*     17 */	add	%o7,%xg10,%xg17


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg8],%f92
/* #00002 */	ldx	[%fp+1479],%xg8


/*     17 */	sxar2
/*     17 */	std,s	%f64,[%xg17]
/*     17 */	add	%xg24,%xg0,%xg18


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f104,%f102,%f124,%f102
/*     17 */	add	%o5,%xg10,%xg20


/*     17 */	sxar2
/*     17 */	std,s	%f66,[%xg20]
/*     17 */	fmaddd,s	%f104,%f116,%f100,%f116


/*     17 */	sxar2
/*     17 */	add	%xg25,%xg10,%xg21
/*     17 */	std,s	%f42,[%xg21]


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f104,%f120,%f118,%f120
/*     17 */	add	%o3,%xg10,%xg22


/*     17 */	sxar2
/*     17 */	add	%xg26,%xg10,%xg10
/*     17 */	std,s	%f36,[%xg22]


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f104,%f112,%f126,%f112
/*     17 */	add	%g3,224,%xg23


/*     17 */	sxar2
/*     17 */	std,s	%f92,[%xg10]
/*     17 */	add	%o2,%xg19,%xg24


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1463],%xg10
/*     17 */	fmaddd,s	%f104,%f60,%f128,%f60


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg24],%f138
/*     17 */	fmaddd,s	%f104,%f114,%f110,%f104


/*     17 */	sxar2
/*     17 */	add	%xg27,%g1,%xg25
/*     17 */	add	%xg28,%g1,%xg26


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg26],%f204
/*     17 */	fmaddd,s	%f48,%f102,%f130,%f102


/*     17 */	sxar2
/*     17 */	add	%xg29,%xg3,%xg27
/*     17 */	ldd,s	[%xg25],%f200


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f48,%f116,%f124,%f116
/*     17 */	ldd,s	[%xg27],%f156


/*     17 */	sxar2
/*     17 */	add	%xg2,672,%xg28
/*     17 */	add	%xg30,%xg0,%xg29


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg29],%f184
/*     17 */	ldd,s	[%xg16],%f180


/*     17 */	sxar2
/*     17 */	add	%xg31,%g1,%xg30
/*     17 */	add	%g2,%g1,%xg31


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg31],%f148
/*     17 */	fmaddd,s	%f72,%f136,%f132,%f134


/*     17 */	sxar2
/*     17 */	add	%xg2,896,%g5
/*     17 */	ldd,s	[%xg30],%f144


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f48,%f60,%f138,%f60
/*     17 */	ldd,s	[%xg7],%f160


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1487],%xg7
/*     17 */	add	%o0,%xg3,%g2


/*     17 */	sxar2
/*     17 */	add	%xg1,%xg3,%o0
/*     17 */	ldd,s	[%o0],%f154


/*     17 */	sxar2
/*     17 */	add	%xg4,%xg0,%xg1
/*     17 */	ldd,s	[%g2],%f152


/*     17 */	sxar2
/*     17 */	add	%xg5,%xg23,%xg4
/*     17 */	std,s	%f102,[%xg4]


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f48,%f120,%f128,%f120
/*     17 */	add	%xg6,%xg23,%xg5


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f48,%f112,%f140,%f112
/*     17 */	std,s	%f116,[%xg5]


/*     17 */	sxar2
/*     17 */	add	%xg7,%xg23,%xg6
/*     17 */	fmuld,s	%f246,%f94,%f142


/*     17 */	sxar2
/*     17 */	add	%xg8,%xg2,%xg7
/*     17 */	add	%xg9,%xg23,%xg8


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg1],%f194
/*     17 */	add	%xg10,%g1,%xg9


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f106,%f148,%f144,%f146
/*     17 */	add	%o7,%xg23,%xg10


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f48,%f104,%f126,%f48
/*     17 */	add	%xg12,%xg0,%xg12


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg9],%f162
/*     17 */	std,s	%f60,[%xg10]


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f40,%f250,%f150,%f250
/*     17 */	add	%xg13,%xg3,%xg13


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f72,%f154,%f152,%f72
/*     17 */	ldd,s	[%xg14],%f198


/*     17 */	sxar2
/*     17 */	add	%o5,%xg23,%xg14
/*     17 */	fmaddd,s	%f244,%f134,%f156,%f134


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg13],%f164
/*     17 */	std,s	%f120,[%xg14]


/*     17 */	sxar2
/*     17 */	fmuld,s	%f86,%f122,%f158
/* #00002 */	ldx	[%fp+1527],%xg14


/*     17 */	sxar2
/*     17 */	add	%xg14,%xg0,%xg16
/*     17 */	fmaddd,s	%f244,%f136,%f132,%f136


/*     17 */	sxar2
/*     17 */	std,s	%f112,[%xg8]
/*     17 */	fsubd,s	%f34,%f160,%f160


/*     17 */	sxar2
/*     17 */	add	%o4,%xg0,%xg17
/*     17 */	add	%o3,%xg23,%xg23


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg17],%f208
/* #00002 */	ldx	[%fp+1551],%xg17


/*     17 */	sxar2
/*     17 */	add	%o2,%g1,%xg20
/*     17 */	fmaddd,s	%f142,%f146,%f162,%f146


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg18],%f168
/* #00002 */	ldx	[%fp+1559],%xg18


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg16],%f188
/* #00002 */	ldx	[%fp+1463],%xg16


/*     17 */	sxar2
/*     17 */	add	%o1,%xg19,%xg19
/*     17 */	ldd,s	[%xg20],%f218


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1503],%xg20
/*     17 */	fmaddd,s	%f244,%f252,%f248,%f252


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg12],%f186
/*     17 */	ldd,s	[%xg19],%f98


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1479],%xg19
/*     17 */	add	%o4,%xg3,%xg21


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f244,%f72,%f164,%f72
/*     17 */	add	%xg16,%xg0,%xg22


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f40,%f134,%f166,%f134
/*     17 */	ldd,s	[%xg21],%f178


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1495],%xg21
/*     17 */	fmaddd,s	%f244,%f154,%f152,%f244


/*     17 */	sxar2
/*     17 */	std,s	%f48,[%xg23]
/*     17 */	add	%xg17,%xg0,%xg23


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f40,%f136,%f156,%f136
/*     17 */	add	%xg18,%xg2,%xg24


/*     17 */	sxar2
/*     17 */	add	%xg19,%xg28,%xg25
/*     17 */	fmaddd,s	%f158,%f172,%f168,%f170


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg22],%f196
/* #00002 */	ldx	[%fp+1519],%xg22


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg24],%f240
/* #00002 */	ldx	[%fp+1471],%xg24


/*     17 */	sxar2
/*     17 */	add	%g4,%g1,%xg26
/*     17 */	fmuld,s	%f246,%f122,%f174


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg7],%f190
/*     17 */	ldd,s	[%xg26],%f234


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1487],%xg26
/*     17 */	fmaddd,s	%f40,%f252,%f96,%f252


/*     17 */	sxar2
/*     17 */	add	%g3,336,%xg27
/*     17 */	ldd,s	[%xg23],%f192


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1567],%xg23
/*     17 */	std,s	%f98,[%xg6]


/*     17 */	sxar2
/*     17 */	fmuld,s	%f86,%f160,%f176
/*     17 */	add	%xg20,%xg27,%xg29


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f40,%f72,%f178,%f72
/*     17 */	std,s	%f134,[%xg29]


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f158,%f184,%f180,%f182
/*     17 */	add	%xg21,%xg27,%xg30


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f40,%f244,%f164,%f40
/*     17 */	std,s	%f136,[%xg30]


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f158,%f188,%f186,%f158
/*     17 */	add	%g4,%xg2,%xg31


/*     17 */	sxar2
/*     17 */	add	%o1,%xg3,%xg3
/*     17 */	add	%xg22,%xg2,%g2


/*     17 */	sxar2
/*     17 */	fsubd,s	%f34,%f190,%f190
/*     17 */	add	%o7,%xg27,%o0


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg3],%f100
/*     17 */	std,s	%f250,[%o0]

/*     17 */	sxar1
/*     17 */	add	%xg23,%xg2,%xg1

/* #00002 */	ldx	[%fp+1543],%o0


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f174,%f170,%f192,%f170
/*     17 */	add	%o5,%xg27,%xg3


/*     17 */	sxar2
/*     17 */	std,s	%f252,[%xg3]
/*     17 */	fmaddd,s	%f174,%f172,%f168,%f172


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1535],%xg3
/*     17 */	add	%xg24,%xg27,%xg4


/*     17 */	sxar2
/*     17 */	std,s	%f72,[%xg4]
/*     17 */	fmaddd,s	%f174,%f188,%f186,%f188


/*     17 */	sxar2
/*     17 */	add	%o3,%xg27,%xg5
/*     17 */	add	%xg26,%xg27,%xg27


/*     17 */	sxar2
/*     17 */	std,s	%f40,[%xg5]
/*     17 */	fmaddd,s	%f174,%f182,%f194,%f182


/*     17 */	sxar2
/*     17 */	add	%g3,448,%xg6
/*     17 */	std,s	%f100,[%xg27]


/*     17 */	sxar2
/*     17 */	add	%o2,%xg0,%xg7
/*     17 */	fmaddd,s	%f174,%f158,%f196,%f158


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg7],%f206
/*     17 */	fmaddd,s	%f174,%f184,%f180,%f174


/*     17 */	sxar2
/*     17 */	add	%xg23,%xg11,%xg8
/*     17 */	add	%xg18,%xg11,%xg9


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg9],%f60
/* #00002 */	ldx	[%fp+1511],%xg9


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f122,%f170,%f198,%f170
/*     17 */	add	%xg17,%g1,%xg10


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg8],%f56
/*     17 */	fmaddd,s	%f122,%f172,%f192,%f172


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg10],%f224
/*     17 */	add	%xg2,1120,%xg8


/*     17 */	sxar2
/*     17 */	add	%o0,%xg2,%xg12
/*     17 */	ldd,s	[%xg12],%f254


/*     17 */	sxar2
/*     17 */	ldd,s	[%g2],%f250
/*     17 */	add	%xg3,%xg11,%xg13


/*     17 */	sxar2
/*     17 */	add	%xg14,%xg11,%xg14
/*     17 */	ldd,s	[%xg14],%f216


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1479],%xg14
/*     17 */	fmaddd,s	%f106,%f204,%f200,%f202


/*     17 */	sxar2
/*     17 */	add	%xg2,1344,%xg19
/*     17 */	ldd,s	[%xg13],%f212


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f122,%f158,%f206,%f158
/*     17 */	ldd,s	[%xg25],%f228


/*     17 */	sxar2
/*     17 */	add	%xg22,%g1,%xg16
/*     17 */	add	%o0,%g1,%xg17


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg17],%f222
/* #00002 */	ldx	[%fp+1463],%xg17


/*     17 */	sxar2
/*     17 */	add	%xg9,%xg2,%xg18
/*     17 */	ldd,s	[%xg16],%f220


/*     17 */	sxar2
/*     17 */	add	%xg20,%xg6,%xg20
/*     17 */	std,s	%f170,[%xg20]


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f122,%f188,%f196,%f188
/*     17 */	add	%xg21,%xg6,%xg21


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f122,%f182,%f208,%f182
/*     17 */	std,s	%f172,[%xg21]


/*     17 */	sxar2
/*     17 */	add	%xg26,%xg6,%xg22
/* #00002 */	ldx	[%fp+1527],%xg21


/*     17 */	sxar2
/*     17 */	fmuld,s	%f246,%f160,%f210
/*     17 */	add	%xg14,%xg15,%xg23


/*     17 */	sxar2
/*     17 */	add	%xg24,%xg6,%xg24
/*     17 */	ldd,s	[%xg18],%f50


/*     17 */	sxar2
/*     17 */	add	%xg17,%xg11,%xg25
/*     17 */	fmaddd,s	%f176,%f216,%f212,%f214


/*     17 */	sxar2
/*     17 */	add	%o7,%xg6,%xg26
/*     17 */	fmaddd,s	%f122,%f174,%f194,%f122


/*     17 */	sxar2
/*     17 */	add	%xg3,%xg2,%xg27
/*     17 */	ldd,s	[%xg25],%f230


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1551],%xg25
/*     17 */	std,s	%f158,[%xg26]


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f94,%f146,%f218,%f146
/* #00002 */	ldx	[%fp+1559],%xg26


/*     17 */	sxar2
/*     17 */	add	%xg9,%g1,%xg29
/*     17 */	fmaddd,s	%f106,%f222,%f220,%f106


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg31],%f54
/*     17 */	add	%o5,%xg6,%xg30


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f142,%f202,%f224,%f202
/*     17 */	ldd,s	[%xg29],%f232


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1503],%xg29
/*     17 */	std,s	%f188,[%xg30]


/*     17 */	sxar2
/*     17 */	fmuld,s	%f86,%f190,%f226
/* #00002 */	ldx	[%fp+1495],%xg30


/*     17 */	sxar2
/*     17 */	add	%xg21,%xg2,%xg31
/*     17 */	fmaddd,s	%f142,%f204,%f200,%f204


/*     17 */	sxar2
/*     17 */	std,s	%f182,[%xg24]
/*     17 */	fsubd,s	%f34,%f228,%f228


/*     17 */	sxar2
/*     17 */	add	%o4,%xg2,%g2
/*     17 */	add	%o3,%xg6,%xg6

/*     17 */	sxar1
/*     17 */	ldd,s	[%g2],%f64

/* #00002 */	ldx	[%fp+1471],%g2


/*     17 */	sxar2
/*     17 */	add	%o2,%xg11,%o0
/*     17 */	fmaddd,s	%f210,%f214,%f230,%f214


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg1],%f236
/*     17 */	ldd,s	[%xg31],%f38


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1519],%xg31
/*     17 */	add	%o1,%xg0,%xg0

/*     17 */	sxar1
/*     17 */	ldd,s	[%o0],%f98

/* #00002 */	ldx	[%fp+1487],%o0


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f142,%f148,%f144,%f148
/*     17 */	ldd,s	[%xg27],%f32


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg0],%f102
/*     17 */	add	%o4,%g1,%xg0


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f142,%f106,%f232,%f106
/*     17 */	add	%xg17,%xg2,%xg1


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f94,%f202,%f234,%f202
/*     17 */	ldd,s	[%xg0],%f248


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f142,%f222,%f220,%f142
/*     17 */	std,s	%f122,[%xg6]


/*     17 */	sxar2
/*     17 */	add	%xg25,%xg2,%xg3
/*     17 */	fmaddd,s	%f94,%f204,%f224,%f204


/*     17 */	sxar2
/*     17 */	add	%xg26,%xg15,%xg4
/*     17 */	add	%xg14,%xg8,%xg5


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f226,%f240,%f236,%f238
/*     17 */	ldd,s	[%xg1],%f52


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg4],%f120
/* #00002 */	ldx	[%fp+1543],%xg4


/*     17 */	sxar2
/*     17 */	add	%g4,%xg11,%xg6
/*     17 */	fmuld,s	%f246,%f190,%f242


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg23],%f46
/*     17 */	ldd,s	[%xg6],%f114


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1535],%xg6
/*     17 */	fmaddd,s	%f94,%f148,%f162,%f148


/*     17 */	sxar2
/*     17 */	add	%g3,560,%xg7
/*     17 */	ldd,s	[%xg3],%f48


/*     17 */	sxar2
/*     17 */	std,s	%f102,[%xg22]
/*     17 */	fmuld,s	%f86,%f228,%f244


/*     17 */	sxar2
/*     17 */	add	%xg29,%xg7,%xg9
/*     17 */	fmaddd,s	%f94,%f106,%f248,%f106


/*     17 */	sxar2
/*     17 */	std,s	%f202,[%xg9]
/*     17 */	fmaddd,s	%f226,%f254,%f250,%f252


/*     17 */	sxar2
/*     17 */	add	%xg30,%xg7,%xg10
/*     17 */	fmaddd,s	%f94,%f142,%f232,%f94


/*     17 */	sxar2
/*     17 */	std,s	%f204,[%xg10]
/*     17 */	fmaddd,s	%f226,%f38,%f32,%f226

/*     17 */	sxar1
/*     17 */	add	%g4,%xg15,%xg12

/*     17 */	add	%o1,%g1,%g1


/*     17 */	sxar2
/*     17 */	add	%xg31,%xg15,%xg13
/*     17 */	fsubd,s	%f34,%f46,%f46


/*     17 */	sxar2
/*     17 */	add	%o7,%xg7,%xg14
/*     17 */	ldd,s	[%g1],%f104

/* #00002 */	ldx	[%fp+1567],%g1


/*     17 */	sxar2
/*     17 */	std,s	%f146,[%xg14]
/* #00002 */	ldx	[%fp+1503],%xg14


/*     17 */	sxar2
/*     17 */	add	%g1,%xg15,%xg16
/*     17 */	fmaddd,s	%f242,%f238,%f48,%f238


/*     17 */	sxar2
/*     17 */	add	%o5,%xg7,%xg17
/*     17 */	std,s	%f148,[%xg17]


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f242,%f240,%f236,%f240
/* #00002 */	ldx	[%fp+1495],%xg17


/*     17 */	sxar2
/*     17 */	add	%g2,%xg7,%xg18
/*     17 */	std,s	%f106,[%xg18]


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f242,%f38,%f32,%f38
/* #00002 */	ldx	[%fp+1487],%xg18


/*     17 */	sxar2
/*     17 */	add	%o3,%xg7,%xg20
/*     17 */	add	%o0,%xg7,%xg7


/*     17 */	sxar2
/*     17 */	std,s	%f94,[%xg20]
/*     17 */	fmaddd,s	%f242,%f252,%f50,%f252


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1479],%xg20
/*     17 */	add	%g3,672,%xg21


/*     17 */	sxar2
/*     17 */	std,s	%f104,[%xg7]
/*     17 */	add	%o2,%xg2,%xg22


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1527],%xg7
/*     17 */	fmaddd,s	%f242,%f226,%f52,%f226


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg22],%f62
/* #00002 */	ldx	[%fp+1471],%xg22


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f242,%f254,%f250,%f242
/*     17 */	add	%g1,%xg28,%xg23


/*     17 */	sxar2
/*     17 */	add	%xg26,%xg28,%xg24
/*     17 */	ldd,s	[%xg24],%f148


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1535],%xg24
/*     17 */	fmaddd,s	%f190,%f238,%f54,%f238


/*     17 */	sxar2
/*     17 */	add	%xg25,%xg11,%xg25
/*     17 */	ldd,s	[%xg23],%f144


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1463],%xg23
/*     17 */	fmaddd,s	%f190,%f240,%f48,%f240


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg25],%f104
/*     17 */	add	%xg2,1568,%xg3


/*     17 */	sxar2
/*     17 */	add	%xg4,%xg15,%xg26
/*     17 */	ldd,s	[%xg26],%f130


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1527],%xg26
/*     17 */	ldd,s	[%xg13],%f126


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1511],%xg13
/*     17 */	add	%xg6,%xg28,%xg27


/*     17 */	sxar2
/*     17 */	add	%xg7,%xg28,%xg29
/*     17 */	ldd,s	[%xg29],%f96


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1551],%xg29
/*     17 */	fmaddd,s	%f176,%f60,%f56,%f58


/*     17 */	sxar2
/*     17 */	add	%xg2,1792,%xg0
/*     17 */	ldd,s	[%xg27],%f72


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f190,%f226,%f62,%f226
/*     17 */	ldd,s	[%xg5],%f36


/*     17 */	sxar2
/*     17 */	add	%xg31,%xg11,%xg30
/*     17 */	add	%xg4,%xg11,%xg31


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg31],%f102
/* #00002 */	ldx	[%fp+1479],%xg31


/*     17 */	sxar2
/*     17 */	add	%xg13,%xg15,%g1
/*     17 */	ldd,s	[%xg30],%f100


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1559],%xg30
/*     17 */	add	%xg14,%xg21,%g2


/*     17 */	sxar2
/*     17 */	std,s	%f238,[%g2]
/*     17 */	fmaddd,s	%f190,%f38,%f52,%f38

/* #00002 */	ldx	[%fp+1495],%g2


/*     17 */	sxar2
/*     17 */	add	%xg17,%xg21,%o0
/*     17 */	fmaddd,s	%f190,%f252,%f64,%f252


/*     17 */	sxar2
/*     17 */	std,s	%f240,[%o0]
/*     17 */	add	%xg18,%xg21,%xg1

/* #00002 */	ldx	[%fp+1519],%o0


/*     17 */	sxar2
/*     17 */	fmuld,s	%f246,%f228,%f68
/*     17 */	add	%xg20,%g5,%xg4


/*     17 */	sxar2
/*     17 */	add	%xg22,%xg21,%xg5
/*     17 */	ldd,s	[%g1],%f138

/* #00002 */	ldx	[%fp+1503],%g1


/*     17 */	sxar2
/*     17 */	add	%xg23,%xg28,%xg6
/*     17 */	fmaddd,s	%f244,%f96,%f72,%f94


/*     17 */	sxar2
/*     17 */	add	%o7,%xg21,%xg7
/*     17 */	fmaddd,s	%f190,%f242,%f50,%f190


/*     17 */	sxar2
/*     17 */	add	%xg24,%xg15,%xg9
/*     17 */	ldd,s	[%xg6],%f110


/*     17 */	sxar2
/*     17 */	std,s	%f226,[%xg7]
/*     17 */	fmaddd,s	%f160,%f214,%f98,%f214


/*     17 */	sxar2
/*     17 */	add	%xg13,%xg11,%xg10
/*     17 */	fmaddd,s	%f176,%f102,%f100,%f176


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg12],%f142
/*     17 */	add	%o5,%xg21,%xg12


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f210,%f58,%f104,%f58
/*     17 */	ldd,s	[%xg10],%f112


/*     17 */	sxar2
/*     17 */	std,s	%f38,[%xg12]
/*     17 */	fmuld,s	%f86,%f46,%f106


/*     17 */	sxar2
/*     17 */	add	%xg26,%xg15,%xg13
/*     17 */	fmaddd,s	%f210,%f60,%f56,%f60


/*     17 */	sxar2
/*     17 */	std,s	%f252,[%xg5]
/*     17 */	fsubd,s	%f34,%f36,%f36


/*     17 */	sxar2
/*     17 */	add	%o4,%xg15,%xg14
/*     17 */	add	%o3,%xg21,%xg21


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg14],%f152
/*     17 */	add	%o2,%xg28,%xg17


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f68,%f94,%f110,%f94
/*     17 */	ldd,s	[%xg16],%f116


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1527],%xg16
/*     17 */	ldd,s	[%xg13],%f134


/*     17 */	sxar2
/*     17 */	add	%o1,%xg2,%xg18
/*     17 */	ldd,s	[%xg17],%f154


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1519],%xg17
/*     17 */	fmaddd,s	%f210,%f216,%f212,%f216


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg9],%f132
/*     17 */	ldd,s	[%xg18],%f108


/*     17 */	sxar2
/*     17 */	add	%o4,%xg11,%xg20
/*     17 */	fmaddd,s	%f210,%f176,%f112,%f176


/*     17 */	sxar2
/*     17 */	add	%xg23,%xg15,%xg22
/*     17 */	fmaddd,s	%f160,%f58,%f114,%f58


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg20],%f124
/* #00002 */	ldx	[%fp+1511],%xg20


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f210,%f102,%f100,%f210
/*     17 */	std,s	%f190,[%xg21]


/*     17 */	sxar2
/*     17 */	add	%xg29,%xg15,%xg23
/* #00002 */	ldx	[%fp+1503],%xg21


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f160,%f60,%f104,%f60
/*     17 */	add	%xg30,%g5,%xg24


/*     17 */	sxar2
/*     17 */	add	%xg31,%xg3,%xg25
/*     17 */	fmaddd,s	%f106,%f120,%f116,%f118


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg22],%f140
/* #00002 */	ldx	[%fp+1495],%xg22


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg24],%f54
/* #00002 */	ldx	[%fp+1479],%xg24


/*     17 */	sxar2
/*     17 */	add	%g4,%xg28,%xg26
/*     17 */	fmuld,s	%f246,%f46,%f122


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg4],%f32
/*     17 */	ldd,s	[%xg26],%f164


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1463],%xg26
/*     17 */	fmaddd,s	%f160,%f216,%f230,%f216


/*     17 */	sxar2
/*     17 */	add	%g3,784,%xg27
/*     17 */	ldd,s	[%xg23],%f136


/*     17 */	sxar2
/*     17 */	std,s	%f108,[%xg1]
/*     17 */	fmuld,s	%f86,%f36,%f42


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1567],%xg1
/*     17 */	add	%g1,%xg27,%xg29


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f160,%f176,%f124,%f176
/*     17 */	std,s	%f58,[%xg29]


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f106,%f130,%f126,%f128
/*     17 */	add	%g2,%xg27,%xg30


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f160,%f210,%f112,%f160
/*     17 */	std,s	%f60,[%xg30]


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f106,%f134,%f132,%f106
/* #00002 */	ldx	[%fp+1511],%xg30


/*     17 */	sxar2
/*     17 */	add	%g4,%g5,%xg31
/*     17 */	add	%o1,%xg11,%xg11

/*     17 */	add	%o0,%g5,%g2


/*     17 */	sxar2
/*     17 */	fsubd,s	%f34,%f32,%f32
/*     17 */	add	%o7,%xg27,%o0


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg11],%f112
/*     17 */	std,s	%f214,[%o0]


/*     17 */	sxar2
/*     17 */	add	%xg1,%g5,%xg1
/*     17 */	fmaddd,s	%f122,%f118,%f136,%f118


/*     17 */	sxar2
/*     17 */	add	%o5,%xg27,%xg4
/*     17 */	std,s	%f216,[%xg4]


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f122,%f120,%f116,%f120
/* #00002 */	ldx	[%fp+1471],%xg4


/*     17 */	sxar2
/*     17 */	add	%xg4,%xg27,%xg5
/*     17 */	std,s	%f176,[%xg5]


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f122,%f134,%f132,%f134
/* #00002 */	ldx	[%fp+1487],%xg5


/*     17 */	sxar2
/*     17 */	add	%o3,%xg27,%xg6
/*     17 */	add	%xg5,%xg27,%xg27


/*     17 */	sxar2
/*     17 */	std,s	%f160,[%xg6]
/*     17 */	fmaddd,s	%f122,%f128,%f138,%f128


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1567],%xg6
/*     17 */	add	%g3,896,%xg7


/*     17 */	sxar2
/*     17 */	std,s	%f112,[%xg27]
/*     17 */	add	%o2,%xg15,%xg9


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f122,%f106,%f140,%f106
/*     17 */	ldd,s	[%xg9],%f150


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1559],%xg9
/*     17 */	fmaddd,s	%f122,%f130,%f126,%f122


/*     17 */	sxar2
/*     17 */	add	%xg6,%xg8,%xg10
/*     17 */	add	%xg9,%xg8,%xg11


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg11],%f88
/* #00002 */	ldx	[%fp+1551],%xg11


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f46,%f118,%f142,%f118
/*     17 */	add	%xg11,%xg28,%xg12


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg10],%f84
/*     17 */	fmaddd,s	%f46,%f120,%f136,%f120


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg12],%f160
/* #00002 */	ldx	[%fp+1543],%xg12


/*     17 */	sxar2
/*     17 */	add	%xg2,2016,%g1
/*     17 */	add	%xg12,%g5,%xg13


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg13],%f60
/* #00002 */	ldx	[%fp+1535],%xg13


/*     17 */	sxar2
/*     17 */	ldd,s	[%g2],%f56
/*     17 */	add	%xg13,%xg8,%xg14


/*     17 */	sxar2
/*     17 */	add	%xg16,%xg8,%xg16
/*     17 */	ldd,s	[%xg16],%f66


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f244,%f148,%f144,%f146
/*     17 */	add	%xg2,2240,%xg2


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg14],%f62
/*     17 */	fmaddd,s	%f46,%f106,%f150,%f106


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg25],%f40
/*     17 */	add	%xg17,%xg28,%xg17


/*     17 */	sxar2
/*     17 */	add	%xg12,%xg28,%xg18
/*     17 */	ldd,s	[%xg18],%f158


/*     17 */	sxar2
/*     17 */	add	%xg20,%g5,%xg20
/*     17 */	ldd,s	[%xg17],%f156


/*     17 */	sxar2
/*     17 */	add	%xg21,%xg7,%xg21
/*     17 */	std,s	%f118,[%xg21]


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f46,%f134,%f140,%f134
/*     17 */	add	%xg22,%xg7,%xg22


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f46,%f128,%f152,%f128
/*     17 */	std,s	%f120,[%xg22]


/*     17 */	sxar2
/*     17 */	add	%xg5,%xg7,%xg23
/*     17 */	fmuld,s	%f246,%f36,%f44


/*     17 */	sxar2
/*     17 */	add	%xg24,%xg19,%xg24
/*     17 */	add	%xg4,%xg7,%xg25


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg20],%f76
/*     17 */	add	%xg26,%xg8,%xg26


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f42,%f66,%f62,%f64
/*     17 */	add	%o7,%xg7,%xg27


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f46,%f122,%f138,%f46
/*     17 */	add	%xg13,%g5,%xg29


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg26],%f78
/*     17 */	std,s	%f106,[%xg27]


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f228,%f94,%f154,%f94
/*     17 */	add	%xg30,%xg28,%xg30


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f244,%f158,%f156,%f244
/*     17 */	ldd,s	[%xg31],%f82


/*     17 */	sxar2
/*     17 */	add	%o5,%xg7,%xg31
/*     17 */	fmaddd,s	%f68,%f146,%f160,%f146


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg30],%f162
/*     17 */	std,s	%f134,[%xg31]


/*     17 */	sxar2
/*     17 */	fmuld,s	%f86,%f32,%f38
/* #00002 */	ldx	[%fp+1527],%xg31


/*     17 */	sxar2
/*     17 */	add	%xg31,%g5,%g2
/*     17 */	fmaddd,s	%f68,%f148,%f144,%f148


/*     17 */	sxar2
/*     17 */	std,s	%f128,[%xg25]
/*     17 */	fsubd,s	%f34,%f40,%f40

/*     17 */	add	%o4,%g5,%o0


/*     17 */	sxar2
/*     17 */	add	%o3,%xg7,%xg7
/*     17 */	ldd,s	[%o0],%f90


/*     17 */	sxar2
/*     17 */	add	%o2,%xg8,%xg4
/*     17 */	fmaddd,s	%f44,%f64,%f78,%f64


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg1],%f166
/*     17 */	ldd,s	[%g2],%f70

/* #00002 */	ldx	[%fp+1463],%g2


/*     17 */	sxar2
/*     17 */	add	%o1,%xg15,%xg15
/*     17 */	ldd,s	[%xg4],%f92


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1479],%xg4
/*     17 */	fmaddd,s	%f68,%f96,%f72,%f96


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg29],%f170
/*     17 */	ldd,s	[%xg15],%f114


/*     17 */	sxar2
/*     17 */	add	%o4,%xg28,%xg5
/*     17 */	fmaddd,s	%f68,%f244,%f162,%f244


/*     17 */	sxar2
/*     17 */	add	%g2,%g5,%xg6
/*     17 */	fmaddd,s	%f228,%f146,%f164,%f146


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg5],%f168
/* #00002 */	ldx	[%fp+1503],%xg5


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f68,%f158,%f156,%f68
/*     17 */	std,s	%f46,[%xg7]


/*     17 */	sxar2
/*     17 */	add	%xg11,%g5,%xg7
/*     17 */	fmaddd,s	%f228,%f148,%f160,%f148


/*     17 */	sxar2
/*     17 */	add	%xg9,%xg19,%xg9
/*     17 */	add	%xg4,%g1,%xg1


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f38,%f54,%f166,%f50
/*     17 */	ldd,s	[%xg6],%f80


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1495],%xg6
/*     17 */	ldd,s	[%xg9],%f116


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1567],%xg9
/*     17 */	add	%g4,%xg8,%xg10


/*     17 */	sxar2
/*     17 */	fmuld,s	%f246,%f32,%f46
/*     17 */	ldd,s	[%xg24],%f48


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg10],%f108
/* #00002 */	ldx	[%fp+1471],%xg10


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f228,%f96,%f110,%f96
/*     17 */	add	%g3,1008,%xg11


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg7],%f74
/* #00002 */	ldx	[%fp+1519],%xg7


/*     17 */	sxar2
/*     17 */	std,s	%f114,[%xg23]
/*     17 */	fmuld,s	%f86,%f40,%f72


/*     17 */	sxar2
/*     17 */	add	%xg5,%xg11,%xg12
/*     17 */	fmaddd,s	%f228,%f244,%f168,%f244


/*     17 */	sxar2
/*     17 */	std,s	%f146,[%xg12]
/*     17 */	fmaddd,s	%f38,%f60,%f56,%f58


/*     17 */	sxar2
/* #00002 */	ldx	[%fp+1487],%xg12
/*     17 */	add	%xg6,%xg11,%xg13


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f228,%f68,%f162,%f228
/*     17 */	std,s	%f148,[%xg13]


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f38,%f70,%f170,%f38
/* #00002 */	ldsw	[%fp+1579],%xg13


/*     17 */	sxar2
/*     17 */	add	%g4,%xg19,%xg5
/*     17 */	add	%o1,%xg28,%xg28


/*     17 */	sxar2
/*     17 */	add	%xg7,%xg19,%xg6
/*     17 */	fsubd,s	%f34,%f48,%f48


/*     17 */	sxar2
/*     17 */	add	%o7,%xg11,%xg14
/*     17 */	ldd,s	[%xg28],%f118


/*     17 */	sxar2
/*     17 */	std,s	%f94,[%xg14]
/*     17 */	add	%xg9,%xg19,%xg7


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f46,%f50,%f74,%f50
/*     17 */	add	%o5,%xg11,%xg15


/*     17 */	sxar2
/*     17 */	std,s	%f96,[%xg15]
/*     17 */	fmaddd,s	%f46,%f54,%f166,%f54


/*     17 */	sxar2
/* #00002 */	ldsw	[%fp+1575],%xg15
/*     17 */	add	%xg10,%xg11,%xg16


/*     17 */	sxar2
/*     17 */	std,s	%f244,[%xg16]
/*     17 */	fmaddd,s	%f46,%f70,%f170,%f70


/*     17 */	sxar2
/*     17 */	add	%o3,%xg11,%xg17
/*     17 */	sub	%xg15,10,%xg16


/*     17 */	sxar2
/*     17 */	add	%xg12,%xg11,%xg11
/*     17 */	cmp	%xg16,19


/*     17 */	sxar2
/*     17 */	std,s	%f228,[%xg17]
/*     17 */	fmaddd,s	%f46,%f58,%f76,%f58

/*     17 */	sxar1
/*     17 */	add	%xg13,10,%xg14

/*     17 */	add	%g3,1120,%g3


/*     17 */	sxar2
/*     17 */	std,s	%f118,[%xg11]
/*     17 */	add	%o2,%g5,%xg9


/*     17 */	sxar2
/* #00002 */	stw	%xg14,[%fp+1579]
/*     17 */	fmaddd,s	%f46,%f38,%f80,%f38

/*     17 */	sxar1
/* #00002 */	stw	%xg16,[%fp+1575]

/*     17 */	bge,pt	%icc, .L4760
	nop


.L4965:


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f32,%f50,%f82,%f50
/*     17 */	fmaddd,s	%f32,%f54,%f74,%f54


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1655],%xg27
/*     17 */	add	%o7,%g3,%xg12


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg9],%f94
/*     17 */	fmaddd,s	%f46,%f60,%f56,%f46


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f32,%f70,%f80,%f70
/*    ??? */	ldx	[%fp+1647],%xg28


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f32,%f58,%f90,%f58
/*    ??? */	ldx	[%fp+1591],%xg29


/*     17 */	sxar2
/*     17 */	add	%o5,%g3,%xg14
/*     17 */	add	%o3,%g3,%xg16


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f42,%f88,%f84,%f86
/*    ??? */	ldx	[%fp+1671],%xg30

/*     17 */	sxar1
/*    ??? */	ldx	[%fp+1583],%xg31

/*     17 */	add	%o1,%g5,%g5


/*     17 */	sxar2
/*     17 */	add	%xg27,%g3,%xg10
/*     17 */	fmaddd,s	%f36,%f64,%f92,%f64


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f44,%f88,%f84,%f88
/*     17 */	ldd,s	[%g5],%f74


/*     17 */	sxar2
/*     17 */	add	%xg28,%g3,%xg11
/*     17 */	fmaddd,s	%f32,%f38,%f94,%f38


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1615],%xg4
/*     17 */	fmaddd,s	%f44,%f66,%f62,%f66


/*     17 */	sxar2
/*     17 */	add	%xg29,%g3,%xg13
/*     17 */	add	%g3,112,%xg20


/*     17 */	sxar2
/*     17 */	std,s	%f50,[%xg10]
/*     17 */	add	%xg30,%xg8,%xg15


/*     17 */	sxar2
/*     17 */	add	%xg31,%g3,%xg17
/*    ??? */	ldx	[%fp+1607],%xg9


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f32,%f46,%f76,%f32
/*     17 */	ldd,s	[%xg15],%f98


/*     17 */	sxar2
/*     17 */	add	%xg27,%xg20,%xg22
/*     17 */	add	%xg28,%xg20,%xg24


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg7],%f112
/*     17 */	add	%xg4,%xg8,%xg18


/*     17 */	sxar2
/*     17 */	add	%o7,%xg20,%xg26
/*    ??? */	ldx	[%fp+1623],%xg31


/*     17 */	sxar2
/*     17 */	add	%o4,%xg8,%xg25
/*     17 */	ldd,s	[%xg18],%f104

/*     17 */	sxar1
/*     17 */	add	%o5,%xg20,%xg27

/*    ??? */	ldx	[%fp+1639],%g5


/*     17 */	sxar2
/*     17 */	add	%xg9,%xg8,%xg21
/*     17 */	fmaddd,s	%f36,%f66,%f78,%f66


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg25],%f124
/*    ??? */	ldd,s	[%fp+1695],%f220


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg21],%f106
/*     17 */	fmaddd,s	%f44,%f86,%f98,%f86


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f36,%f88,%f98,%f88
/*     17 */	std,s	%f54,[%xg11]


/*     17 */	sxar2
/*     17 */	add	%xg29,%xg20,%xg28
/*     17 */	add	%xg30,%xg19,%xg29


/*     17 */	sxar2
/*     17 */	std,s	%f38,[%xg12]
/*     17 */	add	%o3,%xg20,%xg30


/*     17 */	sxar2
/*     17 */	add	%xg31,%xg19,%xg31
/*    ??? */	ldx	[%fp+1599],%xg11


/*     17 */	sxar2
/*     17 */	add	%g5,%xg19,%g5
/*     17 */	add	%o4,%xg19,%xg15


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg29],%f126
/*    ??? */	ldx	[%fp+1583],%xg4


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f42,%f106,%f104,%f42
/*     17 */	fmaddd,s	%f44,%f106,%f104,%f106


/*     17 */	sxar2
/*     17 */	add	%g4,%xg3,%xg29
/*     17 */	fmuld,s	%f220,%f48,%f96

/*     17 */	sxar1
/*    ??? */	ldx	[%fp+1631],%xg9

/*     17 */	add	%g4,%g1,%l1


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f36,%f86,%f108,%f86
/*     17 */	ldd,s	[%xg5],%f134


/*     17 */	sxar2
/*     17 */	add	%xg11,%xg8,%xg23
/*     17 */	add	%o1,%xg8,%xg8


/*     17 */	sxar2
/*     17 */	ldd,s	[%g5],%f122
/*    ??? */	ldd,s	[%fp+1711],%f76


/*     17 */	sxar2
/*     17 */	add	%xg4,%xg20,%xg20
/*     17 */	ldd,s	[%xg23],%f110


/*     17 */	sxar2
/*     17 */	add	%o2,%xg19,%xg11
/*     17 */	std,s	%f70,[%xg14]


/*     17 */	sxar2
/*     17 */	add	%xg9,%xg19,%xg4
/*     17 */	std,s	%f58,[%xg13]


/*     17 */	sxar2
/*     17 */	std,s	%f32,[%xg16]
/*     17 */	fmaddd,s	%f96,%f116,%f112,%f114


/*     17 */	sxar2
/*     17 */	std,s	%f74,[%xg17]
/*     17 */	fmaddd,s	%f44,%f42,%f110,%f44


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f36,%f106,%f110,%f106
/*     17 */	fmuld,s	%f76,%f48,%f100


/*     17 */	sxar2
/*     17 */	fmuld,s	%f76,%f40,%f102
/*     17 */	std,s	%f86,[%xg22]


/*     17 */	sxar2
/*     17 */	std,s	%f88,[%xg24]
/*     17 */	std,s	%f64,[%xg26]


/*     17 */	sxar2
/*     17 */	std,s	%f66,[%xg27]
/*     17 */	ldd,s	[%xg31],%f118


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg8],%f78
/*    ??? */	ldx	[%fp+1607],%xg10


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f36,%f44,%f124,%f36
/*    ??? */	ldx	[%fp+1655],%xg12


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f100,%f116,%f112,%f116
/*     17 */	fmaddd,s	%f100,%f114,%f126,%f114


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg4],%f128
/*     17 */	ldd,s	[%xg6],%f130


/*     17 */	sxar2
/*     17 */	add	%g3,336,%xg4
/*     17 */	add	%o2,%xg3,%xg6


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1647],%xg14
/*     17 */	fmaddd,s	%f96,%f122,%f118,%f120


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f100,%f122,%f118,%f122
/*     17 */	add	%xg10,%xg19,%xg9


/*     17 */	sxar2
/*     17 */	add	%g3,224,%xg10
/*     17 */	ldd,s	[%xg11],%f142


/*     17 */	sxar2
/*     17 */	add	%xg12,%xg10,%xg12
/*    ??? */	ldx	[%fp+1599],%xg13


/*     17 */	sxar2
/*     17 */	add	%o7,%xg10,%xg16
/*     17 */	ldd,s	[%xg9],%f132


/*     17 */	sxar2
/*     17 */	add	%o5,%xg10,%xg18
/*     17 */	add	%o3,%xg10,%xg24


/*     17 */	sxar2
/*     17 */	std,s	%f36,[%xg28]
/*     17 */	add	%xg14,%xg10,%xg14


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f48,%f116,%f126,%f116
/*     17 */	std,s	%f106,[%xg30]


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f48,%f114,%f134,%f114
/*     17 */	std,s	%f78,[%xg20]


/*     17 */	sxar2
/*     17 */	add	%g4,%xg0,%xg30
/*     17 */	add	%xg13,%xg19,%xg13


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg15],%f154
/*     17 */	fmaddd,s	%f100,%f120,%f128,%f120


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f96,%f132,%f130,%f96
/*     17 */	add	%o1,%xg19,%xg19


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg13],%f144
/*     17 */	fmaddd,s	%f48,%f122,%f128,%f122


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f100,%f132,%f130,%f132
/*    ??? */	ldx	[%fp+1679],%xg17


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1663],%xg21
/*     17 */	add	%o4,%xg3,%xg15


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1591],%xg22
/*    ??? */	ldx	[%fp+1671],%xg23


/*     17 */	sxar2
/*     17 */	std,s	%f114,[%xg12]
/*     17 */	std,s	%f116,[%xg14]


/*     17 */	sxar2
/*     17 */	add	%xg17,%xg3,%xg17
/*     17 */	add	%xg21,%xg3,%xg21


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f48,%f120,%f142,%f120
/*     17 */	fmaddd,s	%f100,%f96,%f144,%f100


/*     17 */	sxar2
/*     17 */	add	%xg22,%xg10,%xg22
/*     17 */	add	%xg23,%xg3,%xg23


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg21],%f140
/*     17 */	fmaddd,s	%f48,%f132,%f144,%f132


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg17],%f136
/*     17 */	ldd,s	[%xg23],%f152


/*     17 */	sxar2
/*     17 */	add	%o7,%xg4,%xg17
/*     17 */	add	%o5,%xg4,%xg23


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1623],%xg25
/*    ??? */	ldx	[%fp+1639],%xg26


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1583],%xg27
/*     17 */	ldd,s	[%xg29],%f162


/*     17 */	sxar2
/*     17 */	add	%o3,%xg4,%xg29
/*    ??? */	ldx	[%fp+1631],%xg31


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg19],%f80
/*     17 */	fmaddd,s	%f72,%f140,%f136,%f138


/*     17 */	sxar2
/*     17 */	std,s	%f120,[%xg16]
/*     17 */	fmaddd,s	%f48,%f100,%f154,%f48


/*     17 */	sxar2
/*     17 */	add	%xg25,%xg3,%xg25
/*     17 */	add	%xg26,%xg3,%xg26


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f102,%f140,%f136,%f140
/*     17 */	std,s	%f122,[%xg18]


/*     17 */	sxar2
/*     17 */	add	%xg27,%xg10,%xg10
/*     17 */	ldd,s	[%xg26],%f150


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg25],%f146
/*     17 */	add	%o2,%xg0,%xg18


/*     17 */	sxar2
/*     17 */	add	%xg31,%xg3,%xg31
/*    ??? */	ldx	[%fp+1615],%xg7

/*     17 */	sxar1
/*    ??? */	ldx	[%fp+1607],%xg8

/*    ??? */	ldx	[%fp+1655],%g5


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg31],%f156
/*    ??? */	ldx	[%fp+1647],%xg9


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg6],%f166
/*     17 */	fmaddd,s	%f72,%f150,%f146,%f148


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f102,%f138,%f152,%f138
/*     17 */	std,s	%f48,[%xg22]


/*     17 */	sxar2
/*     17 */	add	%xg7,%xg3,%xg7
/*     17 */	add	%xg8,%xg3,%xg8


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f40,%f140,%f152,%f140
/*     17 */	std,s	%f132,[%xg24]


/*     17 */	sxar2
/*     17 */	add	%g5,%xg4,%xg5
/*     17 */	fmaddd,s	%f102,%f150,%f146,%f150


/*     17 */	sxar2
/*     17 */	std,s	%f80,[%xg10]
/*     17 */	add	%xg9,%xg4,%xg9


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg8],%f160
/*     17 */	ldd,s	[%xg7],%f158


/*     17 */	sxar2
/*     17 */	add	%o4,%xg0,%xg7
/*    ??? */	ldx	[%fp+1687],%xg11


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1599],%xg13
/*    ??? */	ldx	[%fp+1663],%xg21


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg15],%f174
/*    ??? */	ldx	[%fp+1679],%xg26


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f102,%f148,%f156,%f148
/*     17 */	fmaddd,s	%f40,%f138,%f162,%f138


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1591],%xg28
/*     17 */	fmaddd,s	%f72,%f160,%f158,%f72


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f102,%f160,%f158,%f160
/*     17 */	add	%xg11,%xg0,%xg11


/*     17 */	sxar2
/*     17 */	add	%xg13,%xg3,%xg13
/*    ??? */	ldx	[%fp+1671],%xg27


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f40,%f150,%f156,%f150
/*     17 */	add	%xg21,%xg0,%xg21


/*     17 */	sxar2
/*     17 */	add	%o1,%xg3,%xg3
/*     17 */	ldd,s	[%xg11],%f164


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg13],%f168
/*     17 */	add	%xg26,%xg0,%xg26


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg21],%f180
/*     17 */	add	%xg28,%xg4,%xg28


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1623],%xg31
/*     17 */	ldd,s	[%xg26],%f176


/*     17 */	sxar2
/*     17 */	add	%xg27,%xg0,%xg27
/*     17 */	fmaddd,s	%f40,%f148,%f166,%f148


/*     17 */	sxar2
/*     17 */	std,s	%f138,[%xg5]
/*     17 */	add	%o2,%g1,%xg5


/*     17 */	sxar2
/*     17 */	fsubd,s	%f34,%f164,%f164
/*     17 */	fmaddd,s	%f102,%f72,%f168,%f102


/*     17 */	sxar2
/*     17 */	std,s	%f140,[%xg9]
/*     17 */	ldd,s	[%xg27],%f192


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f40,%f160,%f168,%f160
/*     17 */	add	%xg31,%xg0,%xg31

/*    ??? */	ldx	[%fp+1639],%g5


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1583],%xg6
/*    ??? */	ldx	[%fp+1631],%xg8


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg30],%f200
/*     17 */	ldd,s	[%xg31],%f182


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg3],%f82
/*     17 */	std,s	%f148,[%xg17]


/*     17 */	sxar2
/*     17 */	add	%o4,%g1,%xg17
/*     17 */	add	%g5,%xg0,%g5


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1615],%xg12
/*     17 */	fmuld,s	%f220,%f164,%f170


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f40,%f102,%f174,%f40
/*     17 */	add	%xg6,%xg4,%xg4


/*     17 */	sxar2
/*     17 */	add	%xg8,%xg0,%xg6
/*     17 */	fmuld,s	%f76,%f164,%f172


/*     17 */	sxar2
/*     17 */	std,s	%f150,[%xg23]
/*    ??? */	ldx	[%fp+1607],%xg14


/*     17 */	sxar2
/*     17 */	ldd,s	[%g5],%f186
/*    ??? */	ldx	[%fp+1655],%xg16


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg6],%f194
/*     17 */	add	%xg12,%xg0,%xg8


/*     17 */	sxar2
/*     17 */	add	%g3,448,%xg12
/*    ??? */	ldx	[%fp+1647],%xg19


/*     17 */	sxar2
/*     17 */	add	%o7,%xg12,%xg25
/*     17 */	add	%o5,%xg12,%xg11


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg8],%f188
/*     17 */	ldd,s	[%xg18],%f202


/*     17 */	sxar2
/*     17 */	add	%xg14,%xg0,%xg14
/*    ??? */	ldx	[%fp+1599],%xg20


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f170,%f180,%f176,%f178
/*     17 */	fmaddd,s	%f170,%f186,%f182,%f184


/*     17 */	sxar2
/*     17 */	add	%xg16,%xg12,%xg16
/*     17 */	fmaddd,s	%f172,%f180,%f176,%f180


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f172,%f186,%f182,%f186
/*     17 */	std,s	%f40,[%xg28]


/*     17 */	sxar2
/*     17 */	add	%xg19,%xg12,%xg19
/*     17 */	add	%o3,%xg12,%xg24


/*     17 */	sxar2
/*     17 */	std,s	%f160,[%xg29]
/*     17 */	std,s	%f82,[%xg4]


/*     17 */	sxar2
/*     17 */	add	%xg20,%xg0,%xg20
/*    ??? */	ldx	[%fp+1591],%xg22


/*     17 */	sxar2
/*     17 */	add	%o1,%xg0,%xg0
/*     17 */	ldd,s	[%xg14],%f190


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1583],%xg10
/*    ??? */	ldx	[%fp+1631],%xg27


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg1],%f196
/*     17 */	fmaddd,s	%f172,%f178,%f192,%f178


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f172,%f184,%f194,%f184
/*     17 */	ldd,s	[%xg20],%f198


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f164,%f180,%f192,%f180
/*     17 */	fmaddd,s	%f164,%f186,%f194,%f186


/*     17 */	sxar2
/*     17 */	add	%xg22,%xg12,%xg22
/*     17 */	ldd,s	[%xg7],%f208


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f170,%f190,%f188,%f170
/*     17 */	fmaddd,s	%f172,%f190,%f188,%f190


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1679],%xg21
/*     17 */	add	%xg10,%xg12,%xg12


/*     17 */	sxar2
/*     17 */	add	%xg27,%g1,%xg10
/*    ??? */	ldx	[%fp+1663],%xg15


/*     17 */	sxar2
/*     17 */	add	%g3,560,%xg27
/*     17 */	fsubd,s	%f34,%f196,%f196


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg10],%f228
/*    ??? */	ldx	[%fp+1623],%xg26

/*    ??? */	ldx	[%fp+1639],%l0


/*     17 */	sxar2
/*     17 */	add	%o7,%xg27,%xg6
/*     17 */	add	%o5,%xg27,%l3


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1607],%xg31
/*     17 */	add	%o3,%xg27,%l4


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f164,%f178,%f200,%f178
/*     17 */	fmaddd,s	%f164,%f184,%f202,%f184


/*     17 */	sxar2
/*     17 */	add	%xg21,%g1,%xg13
/*    ??? */	ldx	[%fp+1671],%xg21

/*     17 */	add	%g3,672,%g3


/*     17 */	sxar2
/*     17 */	add	%xg15,%g1,%xg15
/*     17 */	fmaddd,s	%f172,%f170,%f198,%f172


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f164,%f190,%f198,%f190
/*     17 */	ldd,s	[%xg5],%f234

/*     17 */	sxar1
/*     17 */	add	%xg26,%g1,%xg26

/*     17 */	add	%l0,%g1,%l0


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg15],%f222
/*     17 */	ldd,s	[%xg13],%f216


/*     17 */	sxar2
/*     17 */	add	%xg31,%g1,%xg31
/*     17 */	ldd,s	[%l0],%f214


/*     17 */	sxar2
/*     17 */	fmuld,s	%f220,%f196,%f204
/*     17 */	fmuld,s	%f76,%f196,%f206


/*     17 */	sxar2
/*     17 */	add	%xg21,%g1,%xg21
/*     17 */	ldd,s	[%xg26],%f210


/*     17 */	sxar2
/*     17 */	std,s	%f178,[%xg16]
/*     17 */	std,s	%f180,[%xg19]


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f164,%f172,%f208,%f164
/*     17 */	std,s	%f184,[%xg25]


/*     17 */	sxar2
/*     17 */	std,s	%f186,[%xg11]
/*     17 */	ldd,s	[%xg21],%f230


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f204,%f214,%f210,%f212
/*     17 */	fmaddd,s	%f204,%f222,%f216,%f218


/*     17 */	sxar2
/*     17 */	ldd,s	[%l1],%f236
/*     17 */	ldd,s	[%xg0],%f84


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f206,%f222,%f216,%f222
/*     17 */	fmaddd,s	%f206,%f214,%f210,%f214


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1615],%xg30
/*    ??? */	ldx	[%fp+1599],%xg3


/*     17 */	sxar2
/*    ??? */	ldx	[%fp+1655],%xg9
/*     17 */	ldd,s	[%xg31],%f226


/*     17 */	sxar2
/*     17 */	std,s	%f164,[%xg22]
/*     17 */	std,s	%f190,[%xg24]


/*     17 */	sxar2
/*     17 */	add	%xg30,%g1,%xg30
/*     17 */	add	%xg3,%g1,%xg3


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f206,%f212,%f228,%f212
/*     17 */	fmaddd,s	%f206,%f218,%f230,%f218


/*     17 */	sxar2
/*     17 */	add	%xg9,%xg27,%xg9
/*     17 */	std,s	%f84,[%xg12]


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f196,%f222,%f230,%f222
/*     17 */	fmaddd,s	%f196,%f214,%f228,%f214

/*     17 */	add	%o1,%g1,%g1


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg30],%f224
/*     17 */	ldd,s	[%xg3],%f232

/*    ??? */	ldx	[%fp+1647],%g5


/*     17 */	sxar2
/*     17 */	ldd,s	[%xg17],%f238
/*    ??? */	ldx	[%fp+1591],%xg14

/*     17 */	sxar1
/*    ??? */	ldx	[%fp+1583],%xg23

/*    ??? */	ldsw	[%fp+1579],%o0

/*     17 */	sxar1
/*     17 */	ldd,s	[%g1],%f86

/*    ??? */	ldsw	[%fp+1575],%g2


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f204,%f226,%f224,%f204
/*     17 */	fmaddd,s	%f196,%f218,%f236,%f218


/*     17 */	sxar2
/*     17 */	add	%g5,%xg27,%l2
/*     17 */	mov	%xg2,%g5


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f206,%f226,%f224,%f226
/*     17 */	fmaddd,s	%f196,%f212,%f234,%f212


/*     17 */	sxar2
/*     17 */	add	%xg14,%xg27,%xg14
/*     17 */	add	%xg23,%xg27,%xg27

/*     17 */	add	%o0,6,%o0

/*     17 */	sub	%g2,6,%g2


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f206,%f204,%f232,%f206
/*     17 */	std,s	%f218,[%xg9]


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f196,%f226,%f232,%f226
/*     17 */	std,s	%f222,[%l2]


/*     17 */	sxar2
/*     17 */	std,s	%f212,[%xg6]
/*     17 */	std,s	%f214,[%l3]


/*     17 */	sxar2
/*     17 */	fmaddd,s	%f196,%f206,%f238,%f196
/*     17 */	std,s	%f196,[%xg14]


/*     17 */	sxar2
/*     17 */	std,s	%f226,[%l4]
/*     17 */	std,s	%f86,[%xg27]

.L4959:


.L4958:


.L4961:


/*     23 */	sxar2
/* #00003 */	ldx	[%fp+1687],%xg17
/*     23 */	add	%g4,%g5,%xg23


/*     38 */	sxar2
/*     38 */	add	%o2,%g5,%xg27
/* #00003 */	ldd,s	[%fp+1695],%f64


/*     23 */	sxar2
/* #00003 */	ldx	[%fp+1679],%xg19
/* #00003 */	ldx	[%fp+1663],%xg21

/*     23 */	add	%o4,%g5,%g1


/*     23 */	sxar2
/*     23 */	add	%o1,%g5,%xg4
/* #00003 */	ldx	[%fp+1671],%xg20


/*     38 */	sxar2
/*     38 */	add	%o7,%g3,%xg26
/* #00003 */	ldd,s	[%fp+1711],%f62


/*     23 */	sxar2
/*     23 */	add	%o5,%g3,%xg1
/* #00003 */	ldx	[%fp+1639],%xg25


/*     28 */	sxar2
/* #00003 */	ldx	[%fp+1623],%xg29
/*     28 */	add	%o3,%g3,%xg3

/*     35 */	add	%o0,1,%o0


/*     21 */	sxar2
/*     21 */	add	%xg17,%g5,%xg18
/*     21 */	ldd,s	[%xg23],%f186

/*     35 */	add	%o0,1,%o0


/*     19 */	sxar2
/*     19 */	add	%xg19,%g5,%xg19
/*     19 */	ldd,s	[%xg18],%f172


/*     23 */	sxar2
/*     23 */	add	%xg21,%g5,%xg21
/* #00003 */	ldx	[%fp+1631],%xg28


/*     21 */	sxar2
/*     21 */	add	%xg20,%g5,%xg20
/*     21 */	ldd,s	[%xg19],%f178

/*     19 */	sxar1
/*     19 */	ldd,s	[%xg21],%f182

/*     35 */	subcc	%g2,2,%g2


/*     23 */	sxar2
/*     23 */	ldd,s	[%xg20],%f184
/*     23 */	add	%xg25,%g5,%xg25


/*     23 */	sxar2
/*     23 */	add	%xg29,%g5,%xg29
/* #00003 */	ldx	[%fp+1615],%xg30


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg25],%f192
/*     21 */	ldd,s	[%xg29],%f188


/*     23 */	sxar2
/*     23 */	fsubd,s	%f34,%f172,%f172
/*     23 */	add	%xg28,%g5,%xg28


/*     23 */	sxar2
/* #00003 */	ldx	[%fp+1607],%xg31
/* #00003 */	ldx	[%fp+1655],%xg22


/*     28 */	sxar2
/*     28 */	ldd,s	[%xg28],%f198
/* #00003 */	ldx	[%fp+1647],%xg23


/*     23 */	sxar2
/*     23 */	add	%xg30,%g5,%xg30
/* #00003 */	ldx	[%fp+1599],%xg0


/*     23 */	sxar2
/*     23 */	ldd,s	[%xg30],%f194
/*     23 */	add	%xg31,%g5,%xg31


/*     23 */	sxar2
/*     23 */	ldd,s	[%xg27],%f202
/*     23 */	add	%xg22,%g3,%xg22


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg31],%f196
/*     21 */	ldd,s	[%g1],%f204


/*     38 */	sxar2
/*     38 */	add	%xg23,%g3,%xg24
/*     38 */	fmuld,s	%f62,%f172,%f174


/*     23 */	sxar2
/*     23 */	fmuld,s	%f64,%f172,%f176
/* #00003 */	ldx	[%fp+1679],%xg8

/*     23 */	sxar1
/*     23 */	add	%xg0,%g5,%xg0

/*     21 */	add	%g5,224,%g5


/*     21 */	sxar2
/* #00003 */	ldx	[%fp+1663],%xg10
/*     21 */	ldd,s	[%xg0],%f200


/*     23 */	sxar2
/*     23 */	add	%xg17,%g5,%xg7
/*     23 */	add	%g4,%g5,%xg11


/*     19 */	sxar2
/* #00003 */	ldx	[%fp+1591],%xg2
/*     19 */	ldd,s	[%xg7],%f206


/*     23 */	sxar2
/*     23 */	add	%o2,%g5,%xg15
/*     23 */	add	%o4,%g5,%xg20


/*     23 */	sxar2
/*     23 */	add	%xg8,%g5,%xg8
/* #00003 */	ldx	[%fp+1671],%xg9


/*     23 */	sxar2
/*     23 */	add	%o1,%g5,%xg25
/*     23 */	add	%xg10,%g5,%xg10


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg8],%f212
/*     21 */	fmaddd,s	%f176,%f182,%f178,%f180


/*     19 */	sxar2
/*     19 */	fmaddd,s	%f174,%f182,%f178,%f182
/*     19 */	ldd,s	[%xg10],%f216


/*     21 */	sxar2
/*     21 */	add	%xg2,%g3,%xg2
/*     21 */	fmaddd,s	%f176,%f192,%f188,%f190


/*     21 */	sxar2
/* #00003 */	ldx	[%fp+1583],%xg5
/*     21 */	fmaddd,s	%f176,%f196,%f194,%f176


/*     19 */	sxar2
/*     19 */	fmaddd,s	%f174,%f192,%f188,%f192
/*     19 */	ldd,s	[%xg4],%f66


/*     23 */	sxar2
/*     23 */	fmaddd,s	%f174,%f196,%f194,%f196
/*     23 */	add	%xg9,%g5,%xg9


/*     35 */	sxar2
/*     35 */	ldd,s	[%xg9],%f218
/*     35 */	fsubd,s	%f34,%f206,%f206


/*     21 */	sxar2
/* #00003 */	ldx	[%fp+1655],%xg6
/*     21 */	ldd,s	[%xg11],%f220

/*     34 */	sxar1
/*     34 */	add	%xg5,%g3,%xg5

/*     35 */	add	%g3,112,%g3


/*     21 */	sxar2
/* #00003 */	ldx	[%fp+1639],%xg14
/*     21 */	fmaddd,s	%f174,%f180,%f184,%f180


/*     28 */	sxar2
/*     28 */	fmaddd,s	%f172,%f182,%f184,%f182
/*     28 */	add	%xg23,%g3,%xg12


/*     21 */	sxar2
/* #00003 */	ldx	[%fp+1623],%xg17
/*     21 */	fmaddd,s	%f174,%f190,%f198,%f190


/*     23 */	sxar2
/*     23 */	fmaddd,s	%f174,%f176,%f200,%f174
/*     23 */	add	%o7,%g3,%xg13


/*     21 */	sxar2
/* #00003 */	ldx	[%fp+1631],%xg16
/*     21 */	fmaddd,s	%f172,%f192,%f198,%f192


/*     23 */	sxar2
/*     23 */	fmaddd,s	%f172,%f196,%f200,%f196
/*     23 */	add	%xg6,%g3,%xg6


/*     23 */	sxar2
/* #00003 */	ldx	[%fp+1615],%xg18
/*     23 */	add	%xg14,%g5,%xg14


/*     38 */	sxar2
/*     38 */	fmuld,s	%f62,%f206,%f208
/*     38 */	fmuld,s	%f64,%f206,%f210


/*     23 */	sxar2
/* #00003 */	ldx	[%fp+1607],%xg19
/*     23 */	add	%xg17,%g5,%xg17


/*     23 */	sxar2
/* #00003 */	ldx	[%fp+1599],%xg21
/*     23 */	add	%xg16,%g5,%xg16


/*     23 */	sxar2
/*     23 */	ldd,s	[%xg17],%f222
/* #00003 */	ldx	[%fp+1591],%xg23


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f172,%f180,%f186,%f180
/*     21 */	ldd,s	[%xg16],%f232


/*     21 */	sxar2
/*     21 */	add	%xg18,%g5,%xg18
/*     21 */	fmaddd,s	%f172,%f190,%f202,%f190


/*     23 */	sxar2
/*     23 */	fmaddd,s	%f172,%f174,%f204,%f172
/*     23 */	add	%xg19,%g5,%xg19


/*     23 */	sxar2
/*     23 */	ldd,s	[%xg25],%f68
/*     23 */	add	%xg21,%g5,%xg21

/*     35 */	add	%g5,224,%g5


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f210,%f216,%f212,%f214
/*     21 */	fmaddd,s	%f208,%f216,%f212,%f216


/*     21 */	sxar2
/*     21 */	add	%xg23,%g3,%xg23
/*     21 */	ldd,s	[%xg21],%f234


/*     28 */	sxar2
/*     28 */	std,s	%f180,[%xg22]
/*     28 */	add	%o5,%g3,%xg22


/*     28 */	sxar2
/*     28 */	std,s	%f182,[%xg24]
/*     28 */	add	%o3,%g3,%xg24


/*     21 */	sxar2
/*     21 */	std,s	%f190,[%xg26]
/*     21 */	std,s	%f192,[%xg1]


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f208,%f214,%f218,%f214
/*     21 */	fmaddd,s	%f206,%f216,%f218,%f216


/*     21 */	sxar2
/*     21 */	std,s	%f172,[%xg2]
/*     21 */	std,s	%f196,[%xg3]


/*     19 */	sxar2
/*     19 */	std,s	%f66,[%xg5]
/*     19 */	ldd,s	[%xg14],%f226


/*     34 */	sxar2
/*     34 */	ldd,s	[%xg18],%f228
/* #00003 */	ldx	[%fp+1583],%xg26


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f206,%f214,%f220,%f214
/*     21 */	fmaddd,s	%f210,%f226,%f222,%f224


/*     34 */	sxar2
/*     34 */	fmaddd,s	%f208,%f226,%f222,%f226
/*     34 */	add	%xg26,%g3,%xg26



/*     21 */	sxar2
/*     21 */	std,s	%f214,[%xg6]
/*     21 */	std,s	%f216,[%xg12]


/*     19 */	sxar2
/*     19 */	ldd,s	[%xg15],%f236
/*     19 */	ldd,s	[%xg19],%f230


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f208,%f224,%f232,%f224
/*     21 */	fmaddd,s	%f206,%f226,%f232,%f226


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg20],%f238
/*     21 */	fmaddd,s	%f210,%f230,%f228,%f210


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f208,%f230,%f228,%f230
/*     21 */	fmaddd,s	%f206,%f224,%f236,%f224


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f208,%f210,%f234,%f208
/*     21 */	fmaddd,s	%f206,%f230,%f234,%f230


/*     21 */	sxar2
/*     21 */	std,s	%f224,[%xg13]
/*     21 */	std,s	%f226,[%xg22]


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f206,%f208,%f238,%f206
/*     21 */	std,s	%f206,[%xg23]


/*     21 */	sxar2
/*     21 */	std,s	%f230,[%xg24]
/*     21 */	std,s	%f68,[%xg26]

/*     35 */	bpos,pt	%icc, .L4961
/*     35 */	add	%g3,112,%g3


.L4957:


.L4773:

/*     17 */	addcc	%g2,1,%g2

/*     17 */	bneg	.L4768
	nop


.L4774:


/*     17 */	sxar2
/*     17 */	ldx	[%i0+2191],%xg0
/*     17 */	ldx	[%i0+2199],%xg1


/*     17 */	sxar2
/*     17 */	add	%xg0,%g5,%xg0
/*     17 */	add	%xg1,%g3,%xg1

.L4786:


/*     38 */	sxar2
/*     38 */	ldd,s	[%xg0+112],%f240
/* #00001 */	ldd,s	[%fp+1695],%f58


/*     35 */	subcc	%g2,1,%g2


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg0+176],%f250
/*     21 */	ldd,s	[%xg0+128],%f246


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg0+192],%f38
/*     21 */	ldd,s	[%xg0+144],%f32


/*     19 */	sxar2
/* #00001 */	ldd,s	[%fp+1711],%f56
/*     19 */	ldd,s	[%xg0+208],%f42


/*     21 */	sxar2
/*     21 */	fsubd,s	%f34,%f240,%f240
/*     21 */	ldd,s	[%xg0+160],%f40


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg0+64],%f252
/*     21 */	ldd,s	[%xg0+80],%f44


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg0],%f254
/*     21 */	ldd,s	[%xg0+16],%f48


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg0+96],%f46
/*     21 */	ldd,s	[%xg0+32],%f50


/*     35 */	sxar2
/*     35 */	ldd,s	[%xg0+48],%f60
/*     35 */	add	%xg0,224,%xg0


/*     38 */	sxar2
/*     38 */	fmuld,s	%f56,%f240,%f242
/*     38 */	fmuld,s	%f58,%f240,%f244


/*     21 */	sxar2
/*     21 */	std,s	%f60,[%xg1+48]
/*     21 */	fmaddd,s	%f244,%f250,%f246,%f248


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f244,%f38,%f32,%f36
/*     21 */	fmaddd,s	%f242,%f250,%f246,%f250


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f244,%f42,%f40,%f244
/*     21 */	fmaddd,s	%f242,%f38,%f32,%f38


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f242,%f42,%f40,%f42
/*     21 */	fmaddd,s	%f242,%f248,%f252,%f248


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f242,%f36,%f44,%f36
/*     21 */	fmaddd,s	%f240,%f250,%f252,%f250


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f242,%f244,%f46,%f242
/*     21 */	fmaddd,s	%f240,%f38,%f44,%f38


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f240,%f42,%f46,%f42
/*     21 */	fmaddd,s	%f240,%f248,%f254,%f248


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f240,%f36,%f48,%f36
/*     21 */	fmaddd,s	%f240,%f242,%f50,%f240


/*     21 */	sxar2
/*     21 */	std,s	%f250,[%xg1+64]
/*     21 */	std,s	%f38,[%xg1+80]


/*     21 */	sxar2
/*     21 */	std,s	%f42,[%xg1+96]
/*     21 */	std,s	%f248,[%xg1]


/*     21 */	sxar2
/*     21 */	std,s	%f36,[%xg1+16]
/*     21 */	std,s	%f240,[%xg1+32]

/*     35 */	sxar1
/*     35 */	add	%xg1,112,%xg1

/*     35 */	bpos,pt	%icc, .L4786
/*     35 */	add	%o0,1,%o0


.L4775:


.L4768:

/*     35 */
/*     35 */	ba	.L4758
	nop


.L4770:

/*     35 *//*     35 */	call	__mpc_obar
/*     35 */	ldx	[%fp+2199],%o0

/*     35 *//*     35 */	call	__mpc_obar
/*     35 */	ldx	[%fp+2199],%o0


.L4771:

/*     35 */	ret
	restore



.LLFE5:
	.size	_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE._OMP_1,.-_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE._OMP_1
	.type	_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE._OMP_1,#function
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite4-k.cpp _ZN7Gravity19calc_force_in_rangeEiidP5Force $"
	.section	".text"
	.global	_ZN7Gravity19calc_force_in_rangeEiidP5Force
	.align	64
_ZN7Gravity19calc_force_in_rangeEiidP5Force:
.LLFB6:
.L603:

/*     38 */	save	%sp,-1520,%sp
.LLCFI4:
/*     38 */	stw	%i2,[%fp+2195]
/*     38 */	stx	%i0,[%fp+2175]
/*     38 */	stw	%i1,[%fp+2187]
/*     38 */	std	%f6,[%fp+2199]
/*     38 */	stx	%i4,[%fp+2207]

.L604:

/*     43 */	sethi	%h44(.LB0..113.1),%l0

/*     43 */	or	%l0,%m44(.LB0..113.1),%l0

/*     43 */	sllx	%l0,12,%l0

/*     43 */	or	%l0,%l44(.LB0..113.1),%l0


/*     43 */	sxar2
/*     43 */	ldsb	[%l0],%xg0
/*     43 */	cmp	%xg0,%g0

/*     43 */	bne,pt	%icc, .L606
	nop


.L605:


.LLEHB0:
/*     43 */	call	__cxa_guard_acquire
/*     43 */	mov	%l0,%o0
.LLEHE0:


.L4578:

/*     43 */	cmp	%o0,%g0

/*     43 */	be	.L606
	nop


.L607:

/*     44 */	sethi	%h44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%o0

/*     44 */	sethi	%h44(_ZN7Gravity6GForceC1Ev),%o3

/*     44 */	or	%o0,%m44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%o0

/*     44 */	or	%o3,%m44(_ZN7Gravity6GForceC1Ev),%o3

/*     44 */	sllx	%o0,12,%o0

/*     44 */	sllx	%o3,12,%o3

/*     44 */	or	%o0,%l44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%o0

/*     44 */	or	%o3,%l44(_ZN7Gravity6GForceC1Ev),%o3

/*     44 */	sethi	%hi(8192),%o1

/*     44 */	mov	96,%o2


.LLEHB1:
/*     44 */	call	__cxa_vec_ctor
/*     44 */	mov	%g0,%o4
.LLEHE1:


.L4730:

/*     44 */	ba	.L4577
	nop


.L610:

/*     44 */

.L611:


/*     44 */	call	__cxa_guard_abort
/*     44 */	mov	%l0,%o0


.L4576:


.LLEHB2:
/*     44 */	call	_Unwind_Resume
/*     44 */	mov	%i0,%o0


.L4577:


/*     44 */	call	__cxa_guard_release
/*     44 */	mov	%l0,%o0
.LLEHE2:


.L606:

/*     46 *//*     46 */	sethi	%h44(_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_2),%o0
/*     46 */	mov	%fp,%l1
/*     46 */	or	%o0,%m44(_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_2),%o0
/*     46 */	mov	%g0,%l2
/*     46 */	sllx	%o0,12,%o0
/*     46 */	mov	%l1,%o1
/*     46 */	or	%o0,%l44(_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_2),%o0
/*     46 */	call	__mpc_opar
/*     46 */	mov	%l2,%o2

/*    106 */
/*    108 *//*    108 */	sethi	%h44(_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_3),%o0
/*    108 */	mov	%l1,%o1
/*    108 */	or	%o0,%m44(_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_3),%o0
/*    108 */	mov	%l2,%o2
/*    108 */	sllx	%o0,12,%o0
/*    108 */	call	__mpc_opar
/*    108 */	or	%o0,%l44(_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_3),%o0

/*    134 */
/*    134 */	ret
	restore



.L666:


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
	.uleb128	.L610-.LLFB6
	.uleb128	0x0
	.uleb128	.LLEHB2-.LLFB6
	.uleb128	.LLEHE2-.LLEHB2
	.uleb128	0x0
	.uleb128	0x0
.LLLSDACSE6:
	.sleb128	0
	.sleb128	0
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite4-k.cpp _ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_2 $"
	.section	".text"
	.align	64
_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_2:
.LLFB7:
.L4794:

/*     46 */	sethi	%hi(4096),%g1
	xor	%g1,-752,%g1
	save	%sp,%g1,%sp
.LLCFI5:
/*     46 */	stx	%i0,[%fp+2175]
/*     46 */	stx	%i1,[%fp+2183]
/*     46 */	stx	%i2,[%fp+2191]
/*     46 */	stx	%i3,[%fp+2199]
/*     46 */	stx	%i0,[%fp+2175]

.L4795:

/*     46 *//*     46 */	sxar1
/*     46 */	ldsw	[%i0+2031],%xg0
/*     46 */
.LLEHB3:
/*     48 */	call	omp_get_thread_num
	nop
/*     48 */	mov	%o0,%l3

.L4796:

/*     49 */
/*     49 */	call	__mpc_pmnm
	nop
/*     49 */	ldx	[%fp+2191],%o7
/*     49 */	cmp	%o7,%o0
/*     49 */	bne,pt	%xcc, .L4802
	nop


.L4797:

/*     49 */
/*     50 */	call	omp_get_num_threads
	nop
.LLEHE3:


.L4798:

/*     50 */	ba	.L4801
	nop


.L4799:


.L4800:

/*      0 */	call	_ZSt9terminatev
	nop


.L4801:

/*     50 */	stw	%o0,[%i0+2027]

.L4802:

/*     50 */
/*     52 */	ldx	[%i0+2175],%o4

/*     53 */	ldsw	[%i0+2187],%l2
/*     53 */	ldsw	[%i0+2195],%o5


/*     53 */	cmp	%l2,%o5
/*     53 */	bge	.L4820
/*     53 */	ldsw	[%o4],%l1


.L4803:


/*     53 */	sxar2
/*     53 */	fzero,s	%f236
/*    ??? */	std,s	%f236,[%fp+-2593]

.L4804:

/*     68 */	sethi	%h44(.LR0.cnt.3),%g1

/*     68 */	sethi	%h44(.LR0.cnt.4),%g2

/*     68 */	or	%g1,%m44(.LR0.cnt.3),%g1

/*     68 */	sxar1
/*     68 */	sethi	%h44(.LR0.cnt.5),%xg0

/*     68 */	or	%g2,%m44(.LR0.cnt.4),%g2

/*     68 */	sxar1
/*     68 */	or	%xg0,%m44(.LR0.cnt.5),%xg0

/*     68 */	sllx	%g1,12,%g1

/*     68 */	sllx	%g2,12,%g2

/*     68 */	or	%g1,%l44(.LR0.cnt.3),%g1

/*     68 */	sxar1
/*     68 */	sllx	%xg0,12,%xg0

/*     68 */	or	%g2,%l44(.LR0.cnt.4),%g2


/*     68 */	sxar2
/*     68 */	ldd	[%g1],%f206
/*     68 */	or	%xg0,%l44(.LR0.cnt.5),%xg0

/*     68 */	srl	%l1,31,%l0

/*     68 */	sra	%l3,%g0,%l3


/*     68 */	sxar2
/*     68 */	ldd	[%g1],%f462
/*     68 */	ldd	[%g2],%f208


/*     68 */	add	%l0,%l1,%l0


/*     68 */	sxar2
/*     68 */	ldd	[%g2],%f464
/*     68 */	ldd	[%xg0],%f210


/*     68 */	sethi	%h44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%l7

/*     68 */	sra	%l0,1,%l0

/*     68 */	add	%l3,%l3,%l4

/*     68 */	sxar1
/*     68 */	ldd	[%xg0],%f466


/*     68 */	add	%l0,%l0,%l5

/*     68 */	or	%l7,%m44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%l7

/*     68 */	add	%l4,%l3,%l4

/*     68 */	mov	1,%i2

/*     68 */	sxar1
/*    ??? */	std,s	%f206,[%fp+-2561]

/*     68 */	sub	%l1,%l5,%l5

/*     68 */	sxar1
/*    ??? */	std,s	%f208,[%fp+-2577]

/*     68 */	sllx	%l7,12,%l7

/*     68 */	sllx	%l4,14,%l4

/*     68 */	add	%l0,1,%l6

/*     68 */	sxar1
/*    ??? */	std,s	%f210,[%fp+-2609]

/*     68 */	or	%l7,%l44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%l7

/*     68 */	sra	%i2,%g0,%i1

.L4805:


/*     60 */	sxar2
/*    ??? */	ldd,s	[%fp+-2593],%f220
/*     60 */	srl	%l2,31,%xg0


/*     60 */	sxar2
/*     60 */	ldd	[%i0+2199],%f478
/*     60 */	add	%xg0,%l2,%xg0


/*     60 */	sxar2
/*     60 */	sra	%xg0,1,%xg0
/*     60 */	sra	%xg0,%g0,%xg0


/*     60 */	sxar2
/*     60 */	std,s	%f220,[%fp+-673]
/*     60 */	sllx	%xg0,3,%xg1


/*     60 */	sxar2
/*     60 */	std,s	%f220,[%fp+-657]
/*     60 */	sub	%xg1,%xg0,%xg1


/*     60 */	sxar2
/*     60 */	std,s	%f220,[%fp+-641]
/*     60 */	sllx	%xg1,4,%xg1


/*     34 */	sxar2
/*     34 */	std,s	%f220,[%fp+-625]
/*     34 */	std,s	%f220,[%fp+-609]


/*     34 */	sxar2
/*     34 */	std,s	%f220,[%fp+-593]
/*     34 */	std,s	%f220,[%fp+-577]


/*     34 */	sxar2
/*     34 */	std,s	%f220,[%fp+-561]
/*     34 */	std,s	%f220,[%fp+-545]


/*     34 */	sxar2
/*     34 */	std,s	%f220,[%fp+-529]
/*     34 */	std,s	%f220,[%fp+-513]


/*     26 */	sxar2
/*     26 */	std,s	%f220,[%fp+-497]
/*     26 */	ldd	[%i0+2199],%f222


/*     60 */	ldx	[%i0+2175],%o3


/*     19 */	sxar2
/*     19 */	ldx	[%o3+16],%xg2
/*     19 */	add	%xg2,%xg1,%xg2


/*     19 */	sxar2
/*     19 */	ldd,s	[%xg2],%f224
/*     19 */	std,s	%f224,[%fp+-481]


/*     19 */	sxar2
/*     19 */	ldd,s	[%xg2+16],%f226
/*     19 */	std,s	%f226,[%fp+-465]


/*     19 */	sxar2
/*     19 */	ldd,s	[%xg2+32],%f228
/*     19 */	std,s	%f228,[%fp+-449]


/*     19 */	sxar2
/*     19 */	ldd,s	[%xg2+64],%f230
/*     19 */	std,s	%f230,[%fp+-433]


/*     19 */	sxar2
/*     19 */	ldd,s	[%xg2+80],%f232
/*     19 */	std,s	%f232,[%fp+-417]


/*     26 */	sxar2
/*     26 */	ldd,s	[%xg2+96],%f234
/*     26 */	std,s	%f222,[%fp+-385]

/*     19 */	sxar1
/*     19 */	std,s	%f234,[%fp+-401]

/*     67 */
/*     67 */
/*     68 */	cmp	%l1,%g0
/*     68 */	ble	.L4818
	nop


.L4806:

/*     68 */	cmp	%l5,%g0

/*     68 */	sxar1
/*     68 */	mov	%l0,%xg4

/*     68 */	be	.L4808
	nop


.L4807:

/*     68 */	sxar1
/*     68 */	mov	%l6,%xg4

.L4808:


/*     68 */	sxar2
/*     68 */	ldx	[%fp+2183],%xg5
/*     68 */	ldx	[%fp+2191],%xg6


/*     68 */	sxar2
/*     68 */	sra	%xg4,%g0,%xg3
/*     68 */	sra	%xg5,%g0,%xg5


/*     68 */	sxar2
/*     68 */	sra	%xg6,%g0,%xg6
/*     68 */	sra	%xg5,%g0,%xg7


/*     68 */	sxar2
/*     68 */	sdivx	%xg3,%xg7,%xg3

/*     68 */	sra	%xg3,%g0,%xg3


/*     68 */	sxar2
/*     68 */	mulx	%xg5,%xg3,%xg5
/*     68 */	subcc	%xg4,%xg5,%xg4

/*     68 */	bne,pt	%icc, .L4810
	nop


.L4809:


/*     68 */	sxar2
/*     68 */	add	%xg6,%xg6,%xg6
/*     68 */	add	%xg3,%xg3,%xg8


/*     68 */	sxar2
/*     68 */	mulx	%xg6,%xg3,%xg6
/*     68 */	add	%xg8,%xg6,%xg8

/*     68 */	sxar1
/*     68 */	sub	%xg8,1,%xg8

/*     68 */	ba	.L4813
	nop


.L4810:

/*     68 */	sxar1
/*     68 */	cmp	%xg6,%xg4

/*     68 */	bl	.L4812
	nop


.L4811:


/*     68 */	sxar2
/*     68 */	mulx	%xg6,%xg3,%xg6
/*     68 */	add	%xg3,%xg3,%xg8


/*     68 */	sxar2
/*     68 */	add	%xg6,%xg4,%xg6
/*     68 */	add	%xg6,%xg6,%xg6


/*     68 */	sxar2
/*     68 */	add	%xg8,%xg6,%xg8
/*     68 */	sub	%xg8,1,%xg8

/*     68 */	ba	.L4813
	nop


.L4812:


/*     68 */	sxar2
/*     68 */	add	%xg3,1,%xg3
/*     68 */	add	%xg6,%xg6,%xg6


/*     68 */	sxar2
/*     68 */	mulx	%xg6,%xg3,%xg6
/*     68 */	add	%xg3,%xg3,%xg8


/*     68 */	sxar2
/*     68 */	add	%xg8,%xg6,%xg8
/*     68 */	sub	%xg8,1,%xg8

.L4813:

/*     68 */	sxar1
/*     68 */	cmp	%xg3,%g0

/*     68 */	be	.L4818
	nop


.L4814:

/*     68 */	sxar1
/*     68 */	sub	%xg8,%xg6,%xg8

/*     68 */	ldx	[%i0+2175],%o2


/*     68 */	sxar2
/*     68 */	ldd,s	[%fp+-513],%f182
/*     68 */	srl	%xg8,31,%xg9


/*     68 */	sxar2
/*     68 */	ldd,s	[%fp+-497],%f184
/*     68 */	add	%xg8,%xg9,%xg8


/*     68 */	sxar2
/*     68 */	ldd,s	[%fp+-545],%f178
/*     68 */	sra	%xg8,1,%xg8


/*     68 */	sxar2
/*     68 */	add	%xg8,1,%xg8
/*     68 */	ldx	[%o2+16],%xg11


/*     68 */	sxar2
/*     68 */	ldd,s	[%fp+-529],%f180
/*     68 */	sra	%xg8,%g0,%xg8


/*     68 */	sxar2
/*     68 */	ldd,s	[%fp+-577],%f174
/*     68 */	sub	%i1,%xg8,%xg8


/*     68 */	sxar2
/*     68 */	ldd,s	[%fp+-561],%f176
/*     68 */	ldd,s	[%fp+-609],%f170


/*     68 */	sxar2
/*     68 */	srax	%xg8,32,%xg10
/*     68 */	ldd,s	[%fp+-593],%f172


/*     68 */	sxar2
/*     68 */	ldd,s	[%fp+-641],%f166
/*     68 */	and	%xg8,%xg10,%xg8


/*     68 */	sxar2
/*     68 */	add	%xg11,48,%xg12
/*     68 */	ldd,s	[%fp+-625],%f168


/*     68 */	sxar2
/*     68 */	ldd,s	[%fp+-673],%f162
/*     68 */	sub	%i2,%xg8,%xg8


/*     68 */	sxar2
/*     68 */	add	%xg11,16,%xg13
/*     68 */	ldd,s	[%fp+-657],%f164


/*     68 */	sxar2
/*     68 */	ldd,s	[%fp+-481],%f34
/*     68 */	cmp	%xg8,9


/*     68 */	sxar2
/*     68 */	add	%xg11,32,%xg14
/*     68 */	ldd,s	[%fp+-465],%f40


/*    195 */	sxar2
/*    195 */	ldd,s	[%fp+-449],%f46
/*    195 */	add	%xg11,64,%xg15


/*     68 */	sxar2
/*     68 */	add	%xg11,80,%xg16
/*     68 */	ldd,s	[%fp+-433],%f124


/*    195 */	sxar2
/*    195 */	ldd,s	[%fp+-417],%f70
/*    195 */	add	%xg11,96,%xg17


/*     68 */	sxar2
/*     68 */	ldd,s	[%fp+-401],%f158
/*     68 */	ldd,s	[%fp+-385],%f64

/*     68 */	bl	.L4970
	nop


.L4966:


.L4973:


/*     68 */	sxar2
/*     68 */	srl	%xg6,31,%xg18
/*     68 */	add	%xg6,2,%xg19


/*     68 */	sxar2
/*    ??? */	ldd,s	[%fp+-2561],%f226
/*     68 */	add	%xg18,%xg6,%xg18


/*     68 */	sxar2
/*     68 */	srl	%xg19,31,%xg20
/*    ??? */	ldd,s	[%fp+-2593],%f122


/*     68 */	sxar2
/*     68 */	sra	%xg18,1,%xg18
/*     68 */	add	%xg20,%xg19,%xg20


/*     68 */	sxar2
/*     68 */	sra	%xg18,%g0,%xg18
/*     68 */	sra	%xg20,1,%xg20


/*     68 */	sxar2
/*     68 */	sllx	%xg18,3,%xg21
/*     68 */	sra	%xg20,%g0,%xg20


/*     68 */	sxar2
/*     68 */	sub	%xg21,%xg18,%xg21
/*     68 */	sllx	%xg20,3,%xg22


/*     68 */	sxar2
/*     68 */	sllx	%xg21,4,%xg21
/*     68 */	sub	%xg22,%xg20,%xg22


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg21+%xg11],%f36
/*     68 */	ldd,s	[%xg21+%xg13],%f42


/*     68 */	sxar2
/*     68 */	sllx	%xg22,4,%xg22
/*     68 */	add	%xg6,4,%xg23


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg21+%xg14],%f48
/*     68 */	ldd,s	[%xg22+%xg11],%f52


/*     68 */	sxar2
/*     68 */	srl	%xg23,31,%xg24
/*     68 */	add	%xg6,6,%xg6


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg22+%xg13],%f56
/*     68 */	ldd,s	[%xg22+%xg14],%f60


/*     68 */	sxar2
/*     68 */	add	%xg24,%xg23,%xg24
/*     68 */	srl	%xg6,31,%xg25


/*     68 */	sxar2
/*     68 */	sra	%xg24,1,%xg24
/*     68 */	ldd,s	[%xg21+%xg16],%f72


/*     68 */	sxar2
/*     68 */	add	%xg25,%xg6,%xg25
/*    ??? */	ldd,s	[%fp+-2577],%f246


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f36,%f226,%f34,%f32
/*     68 */	fmsubd,sc	%f292,%f226,%f34,%f36


/*     68 */	sxar2
/*     68 */	sra	%xg24,%g0,%xg24
/*     68 */	sra	%xg25,1,%xg25


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f42,%f226,%f40,%f38
/*     68 */	fmsubd,sc	%f298,%f226,%f40,%f42


/*     68 */	sxar2
/*     68 */	sllx	%xg24,3,%xg26
/*     68 */	sra	%xg25,%g0,%xg25


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f48,%f226,%f46,%f44
/*     68 */	fmsubd,sc	%f304,%f226,%f46,%f48


/*     68 */	sxar2
/*     68 */	sub	%xg26,%xg24,%xg26
/*     68 */	ldd,s	[%xg21+%xg15],%f126


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f52,%f226,%f34,%f50
/*     68 */	fmsubd,sc	%f308,%f226,%f34,%f52


/*     68 */	sxar2
/*     68 */	sllx	%xg26,4,%xg26
/*     68 */	sllx	%xg25,3,%xg27


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f56,%f226,%f40,%f54
/*     68 */	fmsubd,sc	%f312,%f226,%f40,%f56


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg26+%xg11],%f76
/*     68 */	sub	%xg27,%xg25,%xg27


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f60,%f226,%f46,%f58
/*     68 */	fmsubd,sc	%f316,%f226,%f46,%f60


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg26+%xg13],%f84
/*     68 */	sllx	%xg27,4,%xg27


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f32,%f32,%f64,%f62
/*     68 */	fmaddd,s	%f36,%f36,%f64,%f66


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg26+%xg14],%f88
/*     68 */	fmsubd,sc	%f72,%f226,%f70,%f68


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f328,%f226,%f70,%f72
/*     68 */	fmsubd,sc	%f76,%f226,%f34,%f74


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f332,%f226,%f34,%f76
/*     68 */	fmaddd,s	%f50,%f50,%f64,%f78


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f52,%f52,%f64,%f80
/*     68 */	fmsubd,sc	%f84,%f226,%f40,%f82


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f340,%f226,%f40,%f84
/*     68 */	fmsubd,sc	%f88,%f226,%f46,%f86


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f344,%f226,%f46,%f88
/*     68 */	fmaddd,s	%f38,%f38,%f62,%f62


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f42,%f42,%f66,%f66
/*     68 */	fmaddd,s	%f74,%f74,%f64,%f90


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f76,%f76,%f64,%f92
/*     68 */	fmaddd,s	%f54,%f54,%f78,%f78


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f56,%f56,%f80,%f80
/*     68 */	fmaddd,s	%f44,%f44,%f62,%f62


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f48,%f48,%f66,%f66
/*     68 */	fmaddd,s	%f82,%f82,%f90,%f90


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f84,%f84,%f92,%f92
/*     68 */	fmaddd,s	%f58,%f58,%f78,%f78


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f60,%f60,%f80,%f80
/*     68 */	frsqrtad,s	%f62,%f94


/*     68 */	sxar2
/*     68 */	frsqrtad,s	%f66,%f96
/*     68 */	fmuld,s	%f62,%f246,%f62


/*     68 */	sxar2
/*     68 */	fmuld,s	%f66,%f246,%f66
/*     68 */	frsqrtad,s	%f78,%f98


/*     68 */	sxar2
/*     68 */	frsqrtad,s	%f80,%f100
/*     68 */	fmuld,s	%f78,%f246,%f78


/*     68 */	sxar2
/*     68 */	fmuld,s	%f80,%f246,%f80
/*     68 */	fmuld,s	%f94,%f94,%f102


/*     68 */	sxar2
/*     68 */	fmuld,s	%f96,%f96,%f104
/*     68 */	fmuld,s	%f98,%f98,%f106


/*     68 */	sxar2
/*     68 */	fmuld,s	%f100,%f100,%f108
/*     68 */	fnmsubd,s	%f62,%f102,%f246,%f102


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f66,%f104,%f246,%f104
/*     68 */	fnmsubd,s	%f78,%f106,%f246,%f106


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f80,%f108,%f246,%f108
/*     68 */	fmaddd,s	%f94,%f102,%f94,%f94


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f96,%f104,%f96,%f96
/*     68 */	fmaddd,s	%f98,%f106,%f98,%f98


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f100,%f108,%f100,%f100
/*     68 */	fmuld,s	%f94,%f94,%f110


/*     68 */	sxar2
/*     68 */	fmuld,s	%f96,%f96,%f112
/*     68 */	fnmsubd,s	%f62,%f110,%f246,%f110


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f66,%f112,%f246,%f112
/*     68 */	fmaddd,s	%f94,%f110,%f94,%f94


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f96,%f112,%f96,%f96
/*    ??? */	ldd,s	[%fp+-2609],%f110


/*     68 */	sxar2
/*     68 */	fmuld,s	%f94,%f94,%f114
/*     68 */	fmuld,s	%f96,%f96,%f116


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f62,%f114,%f246,%f62
/*     68 */	fnmsubd,s	%f66,%f116,%f246,%f66

.L4815:


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg27+%xg11],%f238
/*     68 */	fmsubd,sc	%f126,%f226,%f124,%f224


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f86,%f86,%f90,%f90
/*     68 */	fmsubd,sc	%f382,%f226,%f124,%f126


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg21+%xg17],%f242
/*     68 */	fmuld,s	%f98,%f98,%f228


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f88,%f88,%f92,%f92
/*     68 */	fmuld,s	%f100,%f100,%f230


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f94,%f62,%f94,%f94
/*     68 */	fmuld,s	%f38,%f68,%f232


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f96,%f66,%f96,%f96
/*     68 */	fmuld,s	%f42,%f72,%f234


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f238,%f226,%f34,%f236
/*     68 */	fmsubd,sc	%f494,%f226,%f34,%f238


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg27+%xg13],%f102
/*     68 */	fmsubd,sc	%f242,%f226,%f158,%f240


/*     68 */	sxar2
/*     68 */	frsqrtad,s	%f90,%f244
/*     68 */	fmsubd,sc	%f498,%f226,%f158,%f242


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f78,%f228,%f246,%f228
/*     68 */	frsqrtad,s	%f92,%f248


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f80,%f230,%f246,%f230
/*     68 */	fmuld,s	%f94,%f94,%f250


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f32,%f224,%f232,%f232
/*     68 */	fmuld,s	%f96,%f96,%f252


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f36,%f126,%f234,%f234
/*     68 */	fmsubd,sc	%f102,%f226,%f40,%f254


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f358,%f226,%f40,%f102
/*     68 */	ldd,s	[%xg27+%xg14],%f116


/*     68 */	sxar2
/*     68 */	fmuld,s	%f90,%f246,%f90
/*     68 */	fmuld,s	%f244,%f244,%f104


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f98,%f228,%f98,%f98
/*     68 */	fmuld,s	%f92,%f246,%f92


/*     68 */	sxar2
/*     68 */	fmuld,s	%f248,%f248,%f106
/*     68 */	fmaddd,s	%f100,%f230,%f100,%f100


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg21+%xg12],%f128
/*     68 */	fmuld,s	%f110,%f250,%f108


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f44,%f240,%f232,%f232
/*     68 */	fmuld,s	%f110,%f252,%f112


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f48,%f242,%f234,%f234
/*     68 */	fmsubd,sc	%f116,%f226,%f46,%f114


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f372,%f226,%f46,%f116
/*     68 */	ldd,s	[%xg22+%xg16],%f136


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f236,%f236,%f64,%f118
/*     68 */	fmaddd,s	%f238,%f238,%f64,%f120


/*     68 */	sxar2
/*     68 */	fmaddd,sc	%f128,%f94,%f122,%f94
/*     68 */	fmaddd,sc	%f384,%f96,%f122,%f128


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f90,%f104,%f246,%f104
/*     68 */	fmuld,s	%f98,%f98,%f130


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f92,%f106,%f246,%f106
/*     68 */	fmuld,s	%f100,%f100,%f132


/*     68 */	sxar2
/*     68 */	fmuld,s	%f108,%f232,%f108
/*     68 */	fmuld,s	%f112,%f234,%f112


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f136,%f226,%f70,%f134
/*     68 */	fmsubd,sc	%f392,%f226,%f70,%f136


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f254,%f254,%f118,%f118
/*     68 */	fmaddd,s	%f102,%f102,%f120,%f120


/*     68 */	sxar2
/*     68 */	fmuld,s	%f94,%f250,%f94
/*     68 */	fmuld,s	%f128,%f252,%f128


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f244,%f104,%f244,%f244
/*     68 */	fnmsubd,s	%f78,%f130,%f246,%f78


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f248,%f106,%f248,%f248
/*     68 */	fnmsubd,s	%f80,%f132,%f246,%f80


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f112,%f36,%f126,%f126
/*     68 */	fmaddd,s	%f108,%f32,%f224,%f224


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f112,%f42,%f72,%f72
/*     68 */	fmaddd,s	%f108,%f38,%f68,%f68


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f112,%f48,%f242,%f112
/*     68 */	fmaddd,s	%f108,%f44,%f240,%f108


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f128,%f36,%f164,%f164
/*     68 */	fmaddd,s	%f94,%f32,%f162,%f162


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f128,%f42,%f168,%f168
/*     68 */	fmaddd,s	%f94,%f38,%f166,%f166


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f128,%f48,%f172,%f172
/*     68 */	fmaddd,s	%f94,%f44,%f170,%f170


/*     68 */	sxar2
/*     68 */	add	%xg6,2,%xg29
/*     68 */	fmaddd,s	%f128,%f126,%f176,%f126


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f94,%f224,%f174,%f224
/*     68 */	srl	%xg29,31,%xg30


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f128,%f72,%f180,%f72
/*     68 */	fmaddd,s	%f94,%f68,%f178,%f68


/*     68 */	sxar2
/*     68 */	add	%xg30,%xg29,%xg30
/*     68 */	fmaddd,s	%f128,%f112,%f184,%f128


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f94,%f108,%f182,%f94
/*     68 */	sra	%xg30,1,%xg30


/*     68 */	sxar2
/*     68 */	sra	%xg30,%g0,%xg30
/*     68 */	ldd,s	[%xg22+%xg15],%f140


/*     68 */	sxar2
/*     68 */	sllx	%xg30,3,%xg21
/*     68 */	sub	%xg21,%xg30,%xg21


/*     68 */	sxar2
/*     68 */	sllx	%xg21,4,%xg21
/*     68 */	ldd,s	[%xg21+%xg11],%f36


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f140,%f226,%f124,%f138
/*     68 */	fmaddd,s	%f114,%f114,%f118,%f118


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f396,%f226,%f124,%f140
/*     68 */	ldd,s	[%xg22+%xg17],%f152


/*     68 */	sxar2
/*     68 */	fmuld,s	%f244,%f244,%f142
/*     68 */	fmaddd,s	%f116,%f116,%f120,%f120


/*     68 */	sxar2
/*     68 */	fmuld,s	%f248,%f248,%f144
/*     68 */	fmaddd,s	%f98,%f78,%f98,%f98


/*     68 */	sxar2
/*     68 */	fmuld,s	%f54,%f134,%f146
/*     68 */	fmaddd,s	%f100,%f80,%f100,%f100


/*     68 */	sxar2
/*     68 */	fmuld,s	%f56,%f136,%f148
/*     68 */	fmsubd,sc	%f36,%f226,%f34,%f32


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f292,%f226,%f34,%f36
/*     68 */	ldd,s	[%xg21+%xg13],%f42


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f152,%f226,%f158,%f150
/*     68 */	frsqrtad,s	%f118,%f182


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f408,%f226,%f158,%f152
/*     68 */	fnmsubd,s	%f90,%f142,%f246,%f142


/*     68 */	sxar2
/*     68 */	frsqrtad,s	%f120,%f154
/*     68 */	fnmsubd,s	%f92,%f144,%f246,%f144


/*     68 */	sxar2
/*     68 */	fmuld,s	%f98,%f98,%f156
/*     68 */	fmaddd,s	%f50,%f138,%f146,%f146


/*     68 */	sxar2
/*     68 */	fmuld,s	%f100,%f100,%f160
/*     68 */	fmaddd,s	%f52,%f140,%f148,%f148


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f42,%f226,%f40,%f38
/*     68 */	fmsubd,sc	%f298,%f226,%f40,%f42


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg21+%xg14],%f48
/*     68 */	fmuld,s	%f118,%f246,%f118


/*     68 */	sxar2
/*     68 */	fmuld,s	%f182,%f182,%f174
/*     68 */	fmaddd,s	%f244,%f142,%f244,%f244


/*     68 */	sxar2
/*     68 */	fmuld,s	%f120,%f246,%f120
/*     68 */	fmuld,s	%f154,%f154,%f176


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f248,%f144,%f248,%f248
/*     68 */	ldd,s	[%xg22+%xg12],%f184


/*     68 */	sxar2
/*     68 */	fmuld,s	%f110,%f156,%f178
/*     68 */	fmaddd,s	%f58,%f150,%f146,%f146


/*     68 */	sxar2
/*     68 */	fmuld,s	%f110,%f160,%f180
/*     68 */	fmaddd,s	%f60,%f152,%f148,%f148


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f48,%f226,%f46,%f44
/*     68 */	fmsubd,sc	%f304,%f226,%f46,%f48


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg26+%xg16],%f192
/*     68 */	fmaddd,s	%f32,%f32,%f64,%f62


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f36,%f36,%f64,%f66
/*     68 */	fmaddd,sc	%f184,%f98,%f122,%f98


/*     68 */	sxar2
/*     68 */	fmaddd,sc	%f440,%f100,%f122,%f184
/*     68 */	fnmsubd,s	%f118,%f174,%f246,%f174


/*     68 */	sxar2
/*     68 */	fmuld,s	%f244,%f244,%f186
/*     68 */	fnmsubd,s	%f120,%f176,%f246,%f176


/*     68 */	sxar2
/*     68 */	fmuld,s	%f248,%f248,%f188
/*     68 */	fmuld,s	%f178,%f146,%f178


/*     68 */	sxar2
/*     68 */	fmuld,s	%f180,%f148,%f180
/*     68 */	fmsubd,sc	%f192,%f226,%f70,%f190


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f448,%f226,%f70,%f192
/*     68 */	fmaddd,s	%f38,%f38,%f62,%f62


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f42,%f42,%f66,%f66
/*     68 */	fmuld,s	%f98,%f156,%f98


/*     68 */	sxar2
/*     68 */	fmuld,s	%f184,%f160,%f184
/*     68 */	fmaddd,s	%f182,%f174,%f182,%f182


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f90,%f186,%f246,%f90
/*     68 */	fmaddd,s	%f154,%f176,%f154,%f154


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f92,%f188,%f246,%f92
/*     68 */	fmaddd,s	%f180,%f52,%f140,%f140


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f178,%f50,%f138,%f138
/*     68 */	fmaddd,s	%f180,%f56,%f136,%f136


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f178,%f54,%f134,%f134
/*     68 */	fmaddd,s	%f180,%f60,%f152,%f180


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f178,%f58,%f150,%f178
/*     68 */	fmaddd,s	%f184,%f52,%f164,%f164


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f98,%f50,%f162,%f162
/*     68 */	fmaddd,s	%f184,%f56,%f168,%f168


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f98,%f54,%f166,%f166
/*     68 */	fmaddd,s	%f184,%f60,%f172,%f172


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f98,%f58,%f170,%f170
/*     68 */	add	%xg6,4,%xg31


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f184,%f140,%f126,%f140
/*     68 */	fmaddd,s	%f98,%f138,%f224,%f138


/*     68 */	sxar2
/*     68 */	srl	%xg31,31,%g1
/*     68 */	fmaddd,s	%f184,%f136,%f72,%f136


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f98,%f134,%f68,%f134
/*     68 */	add	%g1,%xg31,%g1


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f184,%f180,%f128,%f184
/*     68 */	fmaddd,s	%f98,%f178,%f94,%f98

/*     68 */	sra	%g1,1,%g1

/*     68 */	sra	%g1,%g0,%g1


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg26+%xg15],%f196
/*     68 */	sllx	%g1,3,%xg22


/*     68 */	sxar2
/*     68 */	sub	%xg22,%g1,%xg22
/*     68 */	sllx	%xg22,4,%xg22


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg22+%xg11],%f52
/*     68 */	fmsubd,sc	%f196,%f226,%f124,%f194


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f44,%f44,%f62,%f62
/*     68 */	fmsubd,sc	%f452,%f226,%f124,%f196


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg26+%xg17],%f208
/*     68 */	fmuld,s	%f182,%f182,%f198


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f48,%f48,%f66,%f66
/*     68 */	fmuld,s	%f154,%f154,%f200


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f244,%f90,%f244,%f244
/*     68 */	fmuld,s	%f82,%f190,%f202


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f248,%f92,%f248,%f248
/*     68 */	fmuld,s	%f84,%f192,%f204


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f52,%f226,%f34,%f50
/*     68 */	fmsubd,sc	%f308,%f226,%f34,%f52


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg22+%xg13],%f56
/*     68 */	fmsubd,sc	%f208,%f226,%f158,%f206


/*     68 */	sxar2
/*     68 */	frsqrtad,s	%f62,%f94
/*     68 */	fmsubd,sc	%f464,%f226,%f158,%f208


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f118,%f198,%f246,%f198
/*     68 */	frsqrtad,s	%f66,%f96


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f120,%f200,%f246,%f200
/*     68 */	fmuld,s	%f244,%f244,%f210


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f74,%f194,%f202,%f202
/*     68 */	fmuld,s	%f248,%f248,%f212


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f76,%f196,%f204,%f204
/*     68 */	fmsubd,sc	%f56,%f226,%f40,%f54


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f312,%f226,%f40,%f56
/*     68 */	ldd,s	[%xg22+%xg14],%f60


/*     68 */	sxar2
/*     68 */	fmuld,s	%f62,%f246,%f62
/*     68 */	fmuld,s	%f94,%f94,%f214


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f182,%f198,%f182,%f182
/*     68 */	fmuld,s	%f66,%f246,%f66


/*     68 */	sxar2
/*     68 */	fmuld,s	%f96,%f96,%f216
/*     68 */	fmaddd,s	%f154,%f200,%f154,%f154


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg26+%xg12],%f222
/*     68 */	fmuld,s	%f110,%f210,%f218


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f86,%f206,%f202,%f202
/*     68 */	fmuld,s	%f110,%f212,%f220


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f88,%f208,%f204,%f204
/*     68 */	fmsubd,sc	%f60,%f226,%f46,%f58


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f316,%f226,%f46,%f60
/*     68 */	ldd,s	[%xg27+%xg16],%f180


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f50,%f50,%f64,%f78
/*     68 */	fmaddd,s	%f52,%f52,%f64,%f80


/*     68 */	sxar2
/*     68 */	fmaddd,sc	%f222,%f244,%f122,%f244
/*     68 */	fmaddd,sc	%f478,%f248,%f122,%f222


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f62,%f214,%f246,%f214
/*     68 */	fmuld,s	%f182,%f182,%f224


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f66,%f216,%f246,%f216
/*     68 */	fmuld,s	%f154,%f154,%f228


/*     68 */	sxar2
/*     68 */	fmuld,s	%f218,%f202,%f218
/*     68 */	fmuld,s	%f220,%f204,%f220


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f180,%f226,%f70,%f178
/*     68 */	fmsubd,sc	%f436,%f226,%f70,%f180


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f54,%f54,%f78,%f78
/*     68 */	fmaddd,s	%f56,%f56,%f80,%f80


/*     68 */	sxar2
/*     68 */	fmuld,s	%f244,%f210,%f244
/*     68 */	fmuld,s	%f222,%f212,%f222


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f94,%f214,%f94,%f94
/*     68 */	fnmsubd,s	%f118,%f224,%f246,%f118


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f96,%f216,%f96,%f96
/*     68 */	fnmsubd,s	%f120,%f228,%f246,%f120


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f220,%f76,%f196,%f196
/*     68 */	fmaddd,s	%f218,%f74,%f194,%f194


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f220,%f84,%f192,%f192
/*     68 */	fmaddd,s	%f218,%f82,%f190,%f190


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f220,%f88,%f208,%f220
/*     68 */	fmaddd,s	%f218,%f86,%f206,%f218


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f222,%f76,%f164,%f164
/*     68 */	fmaddd,s	%f244,%f74,%f162,%f162


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f222,%f84,%f168,%f168
/*     68 */	fmaddd,s	%f244,%f82,%f166,%f166


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f222,%f88,%f172,%f172
/*     68 */	fmaddd,s	%f244,%f86,%f170,%f170


/*     68 */	sxar2
/*     68 */	add	%xg6,6,%g2
/*     68 */	fmaddd,s	%f222,%f196,%f140,%f196

/*     68 */	sxar1
/*     68 */	fmaddd,s	%f244,%f194,%f138,%f194

/*     68 */	srl	%g2,31,%g3


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f222,%f192,%f136,%f192
/*     68 */	fmaddd,s	%f244,%f190,%f134,%f190

/*     68 */	add	%g3,%g2,%g3


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f222,%f220,%f184,%f222
/*     68 */	fmaddd,s	%f244,%f218,%f98,%f244

/*     68 */	sra	%g3,1,%g3

/*     68 */	sra	%g3,%g0,%g3


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg27+%xg15],%f176
/*     68 */	sllx	%g3,3,%xg26


/*     68 */	sxar2
/*     68 */	sub	%xg26,%g3,%xg26
/*     68 */	sllx	%xg26,4,%xg26


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg26+%xg11],%f76
/*     68 */	fmsubd,sc	%f176,%f226,%f124,%f174


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f58,%f58,%f78,%f78
/*     68 */	fmsubd,sc	%f432,%f226,%f124,%f176


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg27+%xg17],%f248
/*     68 */	fmuld,s	%f94,%f94,%f230


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f60,%f60,%f80,%f80
/*     68 */	fmuld,s	%f96,%f96,%f232


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f182,%f118,%f182,%f182
/*     68 */	fmuld,s	%f254,%f178,%f234


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f154,%f120,%f154,%f154
/*     68 */	fmuld,s	%f102,%f180,%f240


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f76,%f226,%f34,%f74
/*     68 */	fmsubd,sc	%f332,%f226,%f34,%f76


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg26+%xg13],%f84
/*     68 */	fmsubd,sc	%f248,%f226,%f158,%f242


/*     68 */	sxar2
/*     68 */	frsqrtad,s	%f78,%f98
/*     68 */	fmsubd,sc	%f504,%f226,%f158,%f248


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f62,%f230,%f246,%f230
/*     68 */	frsqrtad,s	%f80,%f100


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f66,%f232,%f246,%f232
/*     68 */	fmuld,s	%f182,%f182,%f250


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f236,%f174,%f234,%f234
/*     68 */	fmuld,s	%f154,%f154,%f252


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f238,%f176,%f240,%f240
/*     68 */	fmsubd,sc	%f84,%f226,%f40,%f82


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f340,%f226,%f40,%f84
/*     68 */	ldd,s	[%xg26+%xg14],%f88


/*     68 */	sxar2
/*     68 */	fmuld,s	%f78,%f246,%f78
/*     68 */	fmuld,s	%f98,%f98,%f104


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f94,%f230,%f94,%f94
/*     68 */	fmuld,s	%f80,%f246,%f80


/*     68 */	sxar2
/*     68 */	fmuld,s	%f100,%f100,%f106
/*     68 */	fmaddd,s	%f96,%f232,%f96,%f96


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg27+%xg12],%f184
/*     68 */	fmuld,s	%f110,%f250,%f108


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f114,%f242,%f234,%f234
/*     68 */	fmuld,s	%f110,%f252,%f112


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f116,%f248,%f240,%f240
/*     68 */	fmsubd,sc	%f88,%f226,%f46,%f86


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f344,%f226,%f46,%f88
/*     68 */	ldd,s	[%xg21+%xg16],%f72


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f74,%f74,%f64,%f90
/*     68 */	fmaddd,s	%f76,%f76,%f64,%f92


/*     68 */	sxar2
/*     68 */	fmaddd,sc	%f184,%f182,%f122,%f182
/*     68 */	fmaddd,sc	%f440,%f154,%f122,%f184


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f78,%f104,%f246,%f104
/*     68 */	fmuld,s	%f94,%f94,%f118


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f80,%f106,%f246,%f106
/*     68 */	fmuld,s	%f96,%f96,%f120


/*     68 */	sxar2
/*     68 */	fmuld,s	%f108,%f234,%f108
/*     68 */	fmuld,s	%f112,%f240,%f112


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f72,%f226,%f70,%f68
/*     68 */	fmsubd,sc	%f328,%f226,%f70,%f72


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f82,%f82,%f90,%f90
/*     68 */	fmaddd,s	%f84,%f84,%f92,%f92


/*     68 */	sxar2
/*     68 */	fmuld,s	%f182,%f250,%f182
/*     68 */	fmuld,s	%f184,%f252,%f184


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f98,%f104,%f98,%f98
/*     68 */	fnmsubd,s	%f62,%f118,%f246,%f62


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f100,%f106,%f100,%f100
/*     68 */	fnmsubd,s	%f66,%f120,%f246,%f66


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f112,%f238,%f176,%f176
/*     68 */	fmaddd,s	%f108,%f236,%f174,%f174


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f112,%f102,%f180,%f180
/*     68 */	fmaddd,s	%f108,%f254,%f178,%f178


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f112,%f116,%f248,%f112
/*     68 */	fmaddd,s	%f108,%f114,%f242,%f108


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f184,%f238,%f164,%f164
/*     68 */	fmaddd,s	%f182,%f236,%f162,%f162


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f184,%f102,%f168,%f168
/*     68 */	fmaddd,s	%f182,%f254,%f166,%f166


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f184,%f116,%f172,%f172
/*     68 */	fmaddd,s	%f182,%f114,%f170,%f170


/*     68 */	sxar2
/*     68 */	add	%xg6,8,%xg6
/*     68 */	fmaddd,s	%f184,%f176,%f196,%f176


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f182,%f174,%f194,%f174
/*     68 */	srl	%xg6,31,%g4


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f184,%f180,%f192,%f180
/*     68 */	fmaddd,s	%f182,%f178,%f190,%f178


/*     68 */	sxar2
/*     68 */	add	%g4,%xg6,%g4
/*     68 */	fmaddd,s	%f184,%f112,%f222,%f184

/*     68 */	sxar1
/*     68 */	fmaddd,s	%f182,%f108,%f244,%f182

/*     68 */	sra	%g4,1,%g4

/*     68 */	sra	%g4,%g0,%g4


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg21+%xg15],%f126
/*     68 */	sllx	%g4,3,%xg27


/*     68 */	sxar2
/*     68 */	sub	%xg27,%g4,%xg27
/*     68 */	sllx	%xg27,4,%xg27


/*     68 */	sxar2
/*     68 */	sub	%xg8,4,%xg8
/*     68 */	cmp	%xg8,10

/*     68 */	bge,pt	%icc, .L4815
	nop


.L4974:


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg27+%xg11],%f132
/*     68 */	fmaddd,s	%f86,%f86,%f90,%f90


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f88,%f88,%f92,%f92
/*    ??? */	ldd,s	[%fp+-2561],%f228


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg27+%xg13],%f136
/*     68 */	ldd,s	[%xg27+%xg14],%f144


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f94,%f62,%f94,%f94
/*     68 */	fmuld,s	%f38,%f68,%f128


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg21+%xg17],%f160
/*    ??? */	ldd,s	[%fp+-2577],%f230


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f96,%f66,%f96,%f96
/*     68 */	fmuld,s	%f42,%f72,%f150


/*     68 */	sxar2
/*     68 */	fmuld,s	%f98,%f98,%f118
/*     68 */	fmuld,s	%f100,%f100,%f120


/*     68 */	sxar2
/*     68 */	add	%xg6,2,%xg6
/*     68 */	ldd,s	[%xg21+%xg12],%f218


/*     68 */	sxar2
/*     68 */	sub	%xg8,4,%xg8
/*     68 */	fmsubd,sc	%f132,%f228,%f34,%f130


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f388,%f228,%f34,%f132
/*    ??? */	ldd,s	[%fp+-2609],%f62


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f136,%f228,%f40,%f134
/*     68 */	fmsubd,sc	%f392,%f228,%f40,%f136


/*     68 */	sxar2
/*    ??? */	ldd,s	[%fp+-2593],%f102
/*     68 */	fmsubd,sc	%f126,%f228,%f124,%f122


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg22+%xg16],%f192
/*     68 */	fmsubd,sc	%f144,%f228,%f46,%f142


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f382,%f228,%f124,%f126
/*     68 */	ldd,s	[%xg22+%xg17],%f226


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f400,%f228,%f46,%f144
/*     68 */	frsqrtad,s	%f90,%f138


/*     68 */	sxar2
/*     68 */	frsqrtad,s	%f92,%f140
/*     68 */	ldd,s	[%xg26+%xg16],%f232


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg22+%xg15],%f204
/*     68 */	fmuld,s	%f90,%f230,%f90


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f160,%f228,%f158,%f156
/*     68 */	ldd,s	[%xg26+%xg15],%f234


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f130,%f130,%f64,%f146
/*     68 */	fmaddd,s	%f132,%f132,%f64,%f148


/*     68 */	sxar2
/*     68 */	fmuld,s	%f92,%f230,%f92
/*     68 */	ldd,s	[%xg26+%xg17],%f240


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f416,%f228,%f158,%f160
/*     68 */	ldd,s	[%xg22+%xg12],%f236


/*     68 */	sxar2
/*     68 */	fmuld,s	%f94,%f94,%f186
/*     68 */	fmaddd,s	%f32,%f122,%f128,%f128


/*     68 */	sxar2
/*     68 */	fmuld,s	%f96,%f96,%f188
/*     68 */	fmaddd,s	%f36,%f126,%f150,%f150


/*     68 */	sxar2
/*     68 */	fmuld,s	%f138,%f138,%f152
/*     68 */	fmuld,s	%f140,%f140,%f154


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f78,%f118,%f230,%f118
/*     68 */	fnmsubd,s	%f80,%f120,%f230,%f120


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg27+%xg16],%f238
/*     68 */	ldd,s	[%xg26+%xg12],%f244


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f134,%f134,%f146,%f146
/*     68 */	fmaddd,s	%f136,%f136,%f148,%f148


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg27+%xg17],%f246
/*     68 */	fmaddd,sc	%f218,%f94,%f102,%f94


/*     68 */	sxar2
/*     68 */	fmaddd,sc	%f474,%f96,%f102,%f218
/*     68 */	ldd,s	[%xg27+%xg12],%f248


/*     68 */	sxar2
/*     68 */	fmuld,s	%f62,%f186,%f194
/*     68 */	fmaddd,s	%f44,%f156,%f128,%f128


/*     68 */	sxar2
/*     68 */	ldd,s	[%xg27+%xg15],%f242
/*     68 */	fmuld,s	%f62,%f188,%f200


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f48,%f160,%f150,%f150
/*     68 */	fnmsubd,s	%f90,%f152,%f230,%f152


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f92,%f154,%f230,%f154
/*     68 */	fmaddd,s	%f98,%f118,%f98,%f98


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f100,%f120,%f100,%f100
/*     68 */	fmaddd,s	%f142,%f142,%f146,%f146


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f144,%f144,%f148,%f148
/*     68 */	fmuld,s	%f94,%f186,%f94


/*     68 */	sxar2
/*     68 */	fmuld,s	%f218,%f188,%f218
/*     68 */	fmsubd,sc	%f192,%f228,%f70,%f190


/*     68 */	sxar2
/*     68 */	fmuld,s	%f194,%f128,%f194
/*     68 */	fmsubd,sc	%f448,%f228,%f70,%f192


/*     68 */	sxar2
/*     68 */	fmuld,s	%f200,%f150,%f200
/*     68 */	fmaddd,s	%f138,%f152,%f138,%f138


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f140,%f154,%f140,%f140
/*     68 */	fmuld,s	%f98,%f98,%f196


/*     68 */	sxar2
/*     68 */	fmuld,s	%f100,%f100,%f198
/*     68 */	frsqrtad,s	%f146,%f210


/*     68 */	sxar2
/*     68 */	frsqrtad,s	%f148,%f212
/*     68 */	fmuld,s	%f146,%f230,%f146


/*     68 */	sxar2
/*     68 */	fmuld,s	%f148,%f230,%f148
/*     68 */	fmaddd,s	%f194,%f32,%f122,%f122


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f194,%f38,%f68,%f68
/*     68 */	fmaddd,s	%f200,%f36,%f126,%f126


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f200,%f42,%f72,%f72
/*     68 */	fmuld,s	%f138,%f138,%f214


/*     68 */	sxar2
/*     68 */	fmuld,s	%f140,%f140,%f216
/*     68 */	fnmsubd,s	%f78,%f196,%f230,%f78


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f80,%f198,%f230,%f80
/*     68 */	fmuld,s	%f210,%f210,%f220


/*     68 */	sxar2
/*     68 */	fmuld,s	%f212,%f212,%f222
/*     68 */	fmaddd,s	%f200,%f48,%f160,%f200


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f194,%f44,%f156,%f194
/*     68 */	fmaddd,s	%f218,%f36,%f164,%f164


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f94,%f32,%f162,%f162
/*     68 */	fmaddd,s	%f218,%f42,%f168,%f168


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f94,%f38,%f166,%f166
/*     68 */	fnmsubd,s	%f90,%f214,%f230,%f214


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f92,%f216,%f230,%f216
/*     68 */	fmaddd,s	%f218,%f48,%f172,%f172


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f94,%f44,%f170,%f170
/*     68 */	fnmsubd,s	%f146,%f220,%f230,%f220


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f148,%f222,%f230,%f222
/*     68 */	fmaddd,s	%f218,%f126,%f176,%f126


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f94,%f122,%f174,%f122
/*     68 */	fmaddd,s	%f218,%f72,%f180,%f72


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f94,%f68,%f178,%f68
/*     68 */	fmaddd,s	%f98,%f78,%f98,%f98


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f100,%f80,%f100,%f100
/*     68 */	fmaddd,s	%f138,%f214,%f138,%f138


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f140,%f216,%f140,%f140
/*     68 */	fmaddd,s	%f218,%f200,%f184,%f218


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f94,%f194,%f182,%f94
/*     68 */	fmaddd,s	%f210,%f220,%f210,%f210


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f212,%f222,%f212,%f212
/*     68 */	fmsubd,sc	%f204,%f228,%f124,%f202


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f232,%f228,%f70,%f250
/*     68 */	fmsubd,sc	%f460,%f228,%f124,%f204


/*     68 */	sxar2
/*     68 */	fmuld,s	%f54,%f190,%f206
/*     68 */	fmuld,s	%f56,%f192,%f208


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f226,%f228,%f158,%f224
/*     68 */	fmuld,s	%f138,%f138,%f78


/*     68 */	sxar2
/*     68 */	fmuld,s	%f140,%f140,%f80
/*     68 */	fmsubd,sc	%f488,%f228,%f70,%f232


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f234,%f228,%f124,%f106
/*     68 */	fmuld,s	%f210,%f210,%f112


/*     68 */	sxar2
/*     68 */	fmuld,s	%f212,%f212,%f114
/*     68 */	fmsubd,sc	%f482,%f228,%f158,%f226


/*     68 */	sxar2
/*     68 */	fmuld,s	%f98,%f98,%f252
/*     68 */	fmaddd,s	%f50,%f202,%f206,%f206


/*     68 */	sxar2
/*     68 */	fmuld,s	%f100,%f100,%f254
/*     68 */	fmaddd,s	%f52,%f204,%f208,%f208


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f490,%f228,%f124,%f234
/*     68 */	fnmsubd,s	%f90,%f78,%f230,%f90


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f92,%f80,%f230,%f92
/*     68 */	fmuld,s	%f82,%f250,%f108


/*     68 */	sxar2
/*     68 */	fmuld,s	%f84,%f232,%f110
/*     68 */	fnmsubd,s	%f146,%f112,%f230,%f112


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f148,%f114,%f230,%f114
/*     68 */	fmsubd,sc	%f240,%f228,%f158,%f116


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f238,%f228,%f70,%f178
/*     68 */	fmuld,s	%f62,%f252,%f66


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f494,%f228,%f70,%f180
/*     68 */	fmaddd,s	%f58,%f224,%f206,%f206


/*     68 */	sxar2
/*     68 */	fmuld,s	%f62,%f254,%f104
/*     68 */	fmaddd,s	%f138,%f90,%f138,%f138


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f140,%f92,%f140,%f140
/*     68 */	fmaddd,s	%f60,%f226,%f208,%f208


/*     68 */	sxar2
/*     68 */	fmaddd,sc	%f236,%f98,%f102,%f98
/*     68 */	fmaddd,s	%f210,%f112,%f210,%f182


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f212,%f114,%f212,%f184
/*     68 */	fmsubd,sc	%f496,%f228,%f158,%f240


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f74,%f106,%f108,%f108
/*     68 */	fmaddd,s	%f76,%f234,%f110,%f110


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f242,%f228,%f124,%f174
/*     68 */	fmsubd,sc	%f498,%f228,%f124,%f176


/*     68 */	sxar2
/*     68 */	fmuld,s	%f134,%f178,%f156
/*     68 */	fmuld,s	%f138,%f138,%f118


/*     68 */	sxar2
/*     68 */	fmuld,s	%f140,%f140,%f120
/*     68 */	fmuld,s	%f136,%f180,%f160


/*     68 */	sxar2
/*     68 */	fmsubd,sc	%f246,%f228,%f158,%f186
/*     68 */	fmuld,s	%f182,%f182,%f128


/*     68 */	sxar2
/*     68 */	fmuld,s	%f184,%f184,%f150
/*     68 */	fmaddd,sc	%f492,%f100,%f102,%f236


/*     68 */	sxar2
/*     68 */	fmuld,s	%f66,%f206,%f66
/*     68 */	fmuld,s	%f104,%f208,%f104


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f86,%f116,%f108,%f108
/*     68 */	fmaddd,s	%f88,%f240,%f110,%f110


/*     68 */	sxar2
/*     68 */	fmaddd,sc	%f244,%f138,%f102,%f138
/*     68 */	fmuld,s	%f62,%f118,%f152


/*     68 */	sxar2
/*     68 */	fmuld,s	%f62,%f120,%f154
/*     68 */	fmsubd,sc	%f502,%f228,%f158,%f246


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f130,%f174,%f156,%f156
/*     68 */	fnmsubd,s	%f146,%f128,%f230,%f146


/*     68 */	sxar2
/*     68 */	fnmsubd,s	%f148,%f150,%f230,%f148
/*     68 */	fmuld,s	%f98,%f252,%f98


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f132,%f176,%f160,%f160
/*     68 */	fmuld,s	%f236,%f254,%f236


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f104,%f52,%f204,%f204
/*     68 */	fmaddd,s	%f66,%f50,%f202,%f202


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f104,%f56,%f192,%f192
/*     68 */	fmaddd,s	%f66,%f54,%f190,%f190


/*     68 */	sxar2
/*     68 */	fmaddd,sc	%f500,%f140,%f102,%f244
/*     68 */	fmuld,s	%f152,%f108,%f152


/*     68 */	sxar2
/*     68 */	fmuld,s	%f154,%f110,%f154
/*     68 */	fmaddd,s	%f182,%f146,%f182,%f182


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f184,%f148,%f184,%f184
/*     68 */	fmaddd,s	%f142,%f186,%f156,%f156


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f144,%f246,%f160,%f160
/*     68 */	fmaddd,s	%f104,%f60,%f226,%f104


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f66,%f58,%f224,%f66
/*     68 */	fmaddd,s	%f236,%f52,%f164,%f164


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f98,%f50,%f162,%f162
/*     68 */	fmaddd,s	%f236,%f56,%f168,%f168


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f98,%f54,%f166,%f166
/*     68 */	fmaddd,s	%f236,%f60,%f172,%f172


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f98,%f58,%f170,%f170
/*     68 */	fmuld,s	%f182,%f182,%f188


/*     68 */	sxar2
/*     68 */	fmuld,s	%f184,%f184,%f194
/*     68 */	fmaddd,s	%f236,%f204,%f126,%f204


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f98,%f202,%f122,%f202
/*     68 */	fmaddd,s	%f236,%f192,%f72,%f192


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f98,%f190,%f68,%f190
/*     68 */	fmuld,s	%f138,%f118,%f138


/*     68 */	sxar2
/*     68 */	fmuld,s	%f244,%f120,%f244
/*     68 */	fmaddd,s	%f154,%f76,%f234,%f234


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f152,%f74,%f106,%f106
/*     68 */	fmaddd,s	%f154,%f84,%f232,%f232


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f152,%f82,%f250,%f250
/*     68 */	fmuld,s	%f62,%f188,%f196


/*     68 */	sxar2
/*     68 */	fmuld,s	%f62,%f194,%f198
/*     68 */	fmaddd,sc	%f248,%f182,%f102,%f182


/*     68 */	sxar2
/*     68 */	fmaddd,sc	%f504,%f184,%f102,%f184
/*     68 */	fmaddd,s	%f236,%f104,%f218,%f236


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f98,%f66,%f94,%f98
/*     68 */	fmaddd,s	%f154,%f88,%f240,%f154


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f152,%f86,%f116,%f152
/*     68 */	fmaddd,s	%f244,%f76,%f164,%f164


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f138,%f74,%f162,%f162
/*     68 */	fmaddd,s	%f244,%f84,%f168,%f168


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f138,%f82,%f166,%f166
/*     68 */	fmuld,s	%f196,%f156,%f196


/*     68 */	sxar2
/*     68 */	fmuld,s	%f198,%f160,%f198
/*     68 */	fmaddd,s	%f244,%f88,%f172,%f172


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f138,%f86,%f170,%f170
/*     68 */	fmaddd,s	%f244,%f234,%f204,%f234


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f138,%f106,%f202,%f106
/*     68 */	fmaddd,s	%f244,%f232,%f192,%f232


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f138,%f250,%f190,%f250
/*     68 */	fmuld,s	%f182,%f188,%f182


/*     68 */	sxar2
/*     68 */	fmuld,s	%f184,%f194,%f184
/*     68 */	fmaddd,s	%f244,%f154,%f236,%f244


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f138,%f152,%f98,%f138
/*     68 */	fmaddd,s	%f198,%f132,%f176,%f176


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f196,%f130,%f174,%f174
/*     68 */	fmaddd,s	%f198,%f136,%f180,%f180


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f196,%f134,%f178,%f178
/*     68 */	fmaddd,s	%f198,%f144,%f246,%f198


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f196,%f142,%f186,%f196
/*     68 */	fmaddd,s	%f184,%f132,%f164,%f164


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f182,%f130,%f162,%f162
/*     68 */	fmaddd,s	%f184,%f136,%f168,%f168


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f182,%f134,%f166,%f166
/*     68 */	fmaddd,s	%f184,%f144,%f172,%f172


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f182,%f142,%f170,%f170
/*     68 */	fmaddd,s	%f184,%f176,%f234,%f176


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f182,%f174,%f106,%f174
/*     68 */	fmaddd,s	%f184,%f180,%f232,%f180


/*     68 */	sxar2
/*     68 */	fmaddd,s	%f182,%f178,%f250,%f178
/*     68 */	fmaddd,s	%f184,%f198,%f244,%f184

/*     68 */	sxar1
/*     68 */	fmaddd,s	%f182,%f196,%f138,%f182

.L4970:


.L4969:


.L4972:


/*    136 */	sxar2
/*    136 */	srl	%xg6,31,%g5
/* #00004 */	ldd,s	[%fp+-2561],%f212


/*     69 */	sxar2
/*     69 */	subcc	%xg8,1,%xg8
/*     69 */	add	%g5,%xg6,%g5


/*    195 */	sxar2
/* #00004 */	ldd,s	[%fp+-2577],%f214
/*    195 */	add	%xg6,2,%xg6

/*     69 */	sra	%g5,1,%g5

/*    153 */	sxar1
/* #00004 */	ldd,s	[%fp+-2593],%f216

/*     69 */	sra	%g5,%g0,%g5

/*     54 */	sxar1
/* #00004 */	ldd,s	[%fp+-2609],%f218

/*     69 */	sllx	%g5,3,%o0

/*     69 */	sub	%o0,%g5,%o0

/*     69 */	sllx	%o0,4,%o0


/*    136 */	sxar2
/*    136 */	ldd,s	[%xg11+%o0],%f126
/*    136 */	ldd,s	[%xg13+%o0],%f130


/*    136 */	sxar2
/*    136 */	ldd,s	[%xg14+%o0],%f134
/*    136 */	ldd,s	[%xg16+%o0],%f142


/*    136 */	sxar2
/*    136 */	ldd,s	[%xg15+%o0],%f138
/*    136 */	ldd,s	[%xg17+%o0],%f150


/*    136 */	sxar2
/*    136 */	ldd,s	[%xg12+%o0],%f192
/*    136 */	fmsubd,sc	%f126,%f212,%f34,%f122


/*    177 */	sxar2
/*    177 */	fmsubd,sc	%f130,%f212,%f40,%f128
/*    177 */	fmsubd,sc	%f382,%f212,%f34,%f126


/*    177 */	sxar2
/*    177 */	fmsubd,sc	%f134,%f212,%f46,%f132
/*    177 */	fmsubd,sc	%f386,%f212,%f40,%f130


/*    136 */	sxar2
/*    136 */	fmsubd,sc	%f390,%f212,%f46,%f134
/*    136 */	fmsubd,sc	%f138,%f212,%f124,%f136


/*    177 */	sxar2
/*    177 */	fmsubd,sc	%f142,%f212,%f70,%f140
/*    177 */	fmsubd,sc	%f394,%f212,%f124,%f138


/*    136 */	sxar2
/*    136 */	fmsubd,sc	%f398,%f212,%f70,%f142
/*    136 */	fmsubd,sc	%f150,%f212,%f158,%f148


/*     44 */	sxar2
/*     44 */	fmsubd,sc	%f406,%f212,%f158,%f150
/*     44 */	fmaddd,s	%f122,%f122,%f64,%f144


/*     54 */	sxar2
/*     54 */	fmaddd,s	%f126,%f126,%f64,%f146
/*     54 */	fmuld,s	%f128,%f140,%f200


/*     44 */	sxar2
/*     44 */	fmuld,s	%f130,%f142,%f202
/*     44 */	fmaddd,s	%f128,%f128,%f144,%f144


/*     54 */	sxar2
/*     54 */	fmaddd,s	%f130,%f130,%f146,%f146
/*     54 */	fmaddd,s	%f122,%f136,%f200,%f200


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f126,%f138,%f202,%f202
/*     44 */	fmaddd,s	%f132,%f132,%f144,%f144


/*     54 */	sxar2
/*     54 */	fmaddd,s	%f134,%f134,%f146,%f146
/*     54 */	fmaddd,s	%f132,%f148,%f200,%f200


/*     60 */	sxar2
/*     60 */	fmaddd,s	%f134,%f150,%f202,%f202
/*     60 */	frsqrtad,s	%f144,%f152


/*     60 */	sxar2
/*     60 */	fmuld,s	%f144,%f214,%f144
/*     60 */	frsqrtad,s	%f146,%f156


/*     32 */	sxar2
/*     32 */	fmuld,s	%f146,%f214,%f146
/*     32 */	fmuld,s	%f152,%f152,%f154


/*     32 */	sxar2
/*     32 */	fmuld,s	%f156,%f156,%f188
/*     32 */	fnmsubd,s	%f144,%f154,%f214,%f154


/*     32 */	sxar2
/*     32 */	fnmsubd,s	%f146,%f188,%f214,%f188
/*     32 */	fmaddd,s	%f152,%f154,%f152,%f152


/*     32 */	sxar2
/*     32 */	fmaddd,s	%f156,%f188,%f156,%f156
/*     32 */	fmuld,s	%f152,%f152,%f160


/*     32 */	sxar2
/*     32 */	fmuld,s	%f156,%f156,%f194
/*     32 */	fnmsubd,s	%f144,%f160,%f214,%f160


/*     32 */	sxar2
/*     32 */	fnmsubd,s	%f146,%f194,%f214,%f194
/*     32 */	fmaddd,s	%f152,%f160,%f152,%f152


/*     32 */	sxar2
/*     32 */	fmaddd,s	%f156,%f194,%f156,%f156
/*     32 */	fmuld,s	%f152,%f152,%f186


/*     32 */	sxar2
/*     32 */	fmuld,s	%f156,%f156,%f196
/*     32 */	fnmsubd,s	%f144,%f186,%f214,%f144


/*     32 */	sxar2
/*     32 */	fnmsubd,s	%f146,%f196,%f214,%f146
/*     32 */	fmaddd,s	%f152,%f144,%f152,%f152


/*     54 */	sxar2
/*     54 */	fmaddd,s	%f156,%f146,%f156,%f156
/*     54 */	fmuld,s	%f152,%f152,%f190


/*     54 */	sxar2
/*     54 */	fmaddd,sc	%f192,%f152,%f216,%f152
/*     54 */	fmuld,s	%f156,%f156,%f198


/*     54 */	sxar2
/*     54 */	fmaddd,sc	%f448,%f156,%f216,%f192
/*     54 */	fmuld,s	%f152,%f190,%f152


/*     54 */	sxar2
/*     54 */	fmuld,s	%f218,%f190,%f190
/*     54 */	fmuld,s	%f218,%f198,%f204


/*     24 */	sxar2
/*     24 */	fmuld,s	%f192,%f198,%f192
/*     24 */	fmaddd,s	%f152,%f122,%f162,%f162


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f152,%f128,%f166,%f166
/*     44 */	fmaddd,s	%f192,%f126,%f164,%f164


/*     54 */	sxar2
/*     54 */	fmaddd,s	%f192,%f130,%f168,%f168
/*     54 */	fmuld,s	%f190,%f200,%f190


/*     24 */	sxar2
/*     24 */	fmuld,s	%f204,%f202,%f204
/*     24 */	fmaddd,s	%f152,%f132,%f170,%f170


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f192,%f134,%f172,%f172
/*     44 */	fmaddd,s	%f204,%f126,%f138,%f126


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f190,%f122,%f136,%f122
/*     44 */	fmaddd,s	%f204,%f130,%f142,%f130


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f190,%f128,%f140,%f128
/*     44 */	fmaddd,s	%f204,%f134,%f150,%f204


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f190,%f132,%f148,%f190
/*     44 */	fmaddd,s	%f192,%f126,%f176,%f176


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f152,%f122,%f174,%f174
/*     44 */	fmaddd,s	%f192,%f130,%f180,%f180


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f152,%f128,%f178,%f178
/*     44 */	fmaddd,s	%f192,%f204,%f184,%f184

/*     24 */	sxar1
/*     24 */	fmaddd,s	%f152,%f190,%f182,%f182

/*    195 */	bne,pt	%icc, .L4972
	nop


.L4968:


/*    195 */	sxar2
/*    195 */	std,s	%f162,[%fp+-673]
/*    195 */	std,s	%f164,[%fp+-657]


/*    195 */	sxar2
/*    195 */	std,s	%f166,[%fp+-641]
/*    195 */	std,s	%f168,[%fp+-625]


/*    195 */	sxar2
/*    195 */	std,s	%f170,[%fp+-609]
/*    195 */	std,s	%f172,[%fp+-593]


/*    195 */	sxar2
/*    195 */	std,s	%f174,[%fp+-577]
/*    195 */	std,s	%f176,[%fp+-561]


/*    195 */	sxar2
/*    195 */	std,s	%f178,[%fp+-545]
/*    195 */	std,s	%f180,[%fp+-529]


/*    195 */	sxar2
/*    195 */	std,s	%f182,[%fp+-513]
/*    195 */	std,s	%f184,[%fp+-497]

.L4818:

/*    101 *//*    101 */	call	__mpc_obar
/*    101 */	ldx	[%fp+2199],%o0

/*    101 */

/*     88 */	sxar2
/*     88 */	add	%l7,%l4,%xg28
/*     88 */	ldd,s	[%fp+-673],%f202
/*     88 */	sxar1
/*     88 */	ldd,s	[%fp+-657],%f200

/*    105 */	add	%l2,2,%l2

/*     88 */	sxar2
/*     88 */	ldd,s	[%fp+-641],%f206
/*     88 */	ldd,s	[%fp+-625],%f204


/*     88 */	sxar2
/*     88 */	ldd,s	[%fp+-609],%f210
/*     88 */	ldd,s	[%fp+-593],%f208
/*     88 */	sxar2
/*     88 */	ldd,s	[%fp+-577],%f214
/*     88 */	ldd,s	[%fp+-561],%f212
/*     88 */	sxar2
/*     88 */	faddd,s	%f202,%f200,%f202
/*     88 */	ldd,s	[%fp+-545],%f218
/*     88 */	sxar2
/*     88 */	ldd,s	[%fp+-529],%f216
/*     88 */	faddd,s	%f206,%f204,%f206
/*     88 */	sxar2
/*     88 */	ldd,s	[%fp+-513],%f222
/*     88 */	faddd,s	%f210,%f208,%f210
/*     88 */	sxar2
/*     88 */	faddd,s	%f214,%f212,%f214
/*     88 */	faddd,s	%f218,%f216,%f218

/*     21 */	sxar2
/*     21 */	ldd,s	[%fp+-497],%f220
/*     21 */	std,s	%f202,[%xg28]
/*     21 */	sxar2
/*     21 */	std,s	%f206,[%xg28+16]
/*     21 */	std,s	%f210,[%xg28+32]


/*     21 */	sxar2
/*     21 */	faddd,s	%f222,%f220,%f222
/*     21 */	std,s	%f214,[%xg28+48]
/*     21 */	sxar2
/*     21 */	std,s	%f218,[%xg28+64]
/*     21 */	std,s	%f222,[%xg28+80]

/*    105 */	ldsw	[%i0+2195],%o1
/*    105 */	cmp	%l2,%o1
/*    105 */	bl,pt	%icc, .L4805
/*    105 */	add	%l4,96,%l4


.L4819:


.L4820:


/*    106 */	call	__mpc_obar
/*    106 */	ldx	[%fp+2199],%o0


.L4821:

/*    106 */	ret
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
	.uleb128	.L4799-.LLFB7
	.uleb128	0x0
.LLLSDACSE7:
	.sleb128	0
	.sleb128	0
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite4-k.cpp _ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_3 $"
	.section	".text"
	.align	64
_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_3:
.LLFB8:
.L4823:

/*    108 */	save	%sp,-400,%sp
.LLCFI6:
/*    108 */	stx	%i0,[%fp+2175]
/*    108 */	stx	%i3,[%fp+2199]
/*    108 */	stx	%i0,[%fp+2175]

.L4824:

/*    108 *//*    108 */	sxar1
/*    108 */	ldsw	[%i0+2035],%xg20
/*    108 */
/*    108 */
/*    108 */
/*    109 */	ldsw	[%i0+2187],%o0
/*    109 */	ldsw	[%i0+2195],%l1
/*    109 */	cmp	%o0,%l1
/*    109 */	bge	.L4837
	nop


.L4825:

/*    109 */	sxar1
/*    109 */	mov	1,%xg19

/*    109 */	sra	%l1,%g0,%l1


/*    109 */	sxar2
/*    109 */	fzero,s	%f72
/*    109 */	stx	%xg19,[%fp+2031]

/*    109 */	sxar1
/*    ??? */	std,s	%f72,[%fp+1871]

.L4826:

/*    109 */	sethi	%h44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%l2

/*    109 */	mov	1,%l7

/*    109 */	or	%l2,%m44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%l2

/*    109 */	sethi	%hi(53184),%i1

/*    109 */	sllx	%l2,12,%l2

/*    109 */	sethi	%hi(102336),%i2

/*    109 */	or	%l2,%l44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%l2

/*    109 */	sethi	%hi(151488),%i3

/*    109 */	add	%fp,2039,%l3

/*    109 */	add	%fp,2023,%l4

/*    109 */	add	%fp,2031,%l5

/*    109 */	sra	%l7,%g0,%l6

/*    109 */	or	%i1,960,%i1

/*    109 */	or	%i2,960,%i2

/*    109 */	or	%i3,960,%i3

/*    109 */	sethi	%hi(196608),%l0

/*    109 */	sethi	%hi(49152),%i4

.L4827:

/*    109 */	sra	%o0,%g0,%o0

/*    109 */	stx	%g0,[%sp+2223]

/*    109 */	mov	2,%o2

/*    109 */	mov	%g0,%o3

/*    109 */	mov	%l1,%o1

/*    109 */	mov	%l3,%o4


/*    109 */	stx	%g0,[%sp+2231]

/*    109 */	stx	%l5,[%sp+2239]


/*    109 */	sxar2
/*    109 */	ldx	[%fp+2199],%xg17
/*    109 */	stx	%xg17,[%sp+2247]

/*    109 */	call	__mpc_ostd_th
/*    109 */	mov	%l4,%o5
/*    109 */	sxar2
/*    109 */	ldx	[%fp+2031],%xg18
/*    109 */	cmp	%xg18,%g0
/*    109 */	ble,pn	%xcc, .L4837
	nop


.L4828:

/*    109 */	ldx	[%fp+2039],%o0


/*    109 */	sxar2
/*    109 */	ldx	[%fp+2023],%xg0
/*    109 */	ldsw	[%i0+2187],%xg7


/*    109 */	sxar2
/*    109 */	ldx	[%i0+2207],%xg8
/*    109 */	ldsw	[%i0+2027],%xg11

/*    109 */	sra	%o0,%g0,%o0


/*    109 */	sxar2
/*    109 */	sra	%xg0,%g0,%xg0
/*    109 */	sub	%xg0,%o0,%xg0


/*    109 */	sxar2
/*    109 */	add	%o0,1,%xg1
/*    109 */	srl	%xg0,31,%xg2


/*    109 */	sxar2
/*    109 */	sra	%o0,%g0,%xg3
/*    109 */	add	%xg0,%xg2,%xg0


/*    109 */	sxar2
/*    109 */	sra	%xg1,%g0,%xg1
/*    109 */	sra	%xg0,1,%xg0


/*    109 */	sxar2
/*    109 */	add	%xg3,%xg3,%xg4
/*    109 */	add	%xg0,1,%xg0


/*    109 */	sxar2
/*    109 */	add	%xg1,%xg1,%xg5
/*    109 */	sra	%xg0,%g0,%xg0


/*    109 */	sxar2
/*    109 */	add	%xg4,%xg3,%xg4
/*    109 */	sub	%l6,%xg0,%xg0


/*    109 */	sxar2
/*    109 */	add	%xg5,%xg1,%xg5
/*    109 */	srax	%xg0,32,%xg6


/*    109 */	sxar2
/*    109 */	sllx	%xg4,4,%xg4
/*    109 */	and	%xg0,%xg6,%xg0


/*    109 */	sxar2
/*    109 */	sllx	%xg5,4,%xg5
/*    109 */	sub	%l7,%xg0,%xg0


/*    109 */	sxar2
/*    109 */	add	%xg8,%xg4,%xg4
/*    109 */	sub	%o0,%xg7,%xg7

/*    109 */	sxar1
/*    109 */	add	%xg8,%xg5,%xg8

.L4829:


/*     25 */	sxar2
/*     25 */	srl	%xg7,31,%xg9
/*    ??? */	ldd,s	[%fp+1871],%f70


/*    110 */	sxar2
/*    110 */	add	%xg9,%xg7,%xg9
/*    110 */	sra	%xg9,1,%xg9


/*     25 */	sxar2
/*     25 */	std,s	%f70,[%fp+1887]
/*     25 */	std,s	%f70,[%fp+1903]


/*     25 */	sxar2
/*     25 */	std,s	%f70,[%fp+1919]
/*     25 */	std,s	%f70,[%fp+1935]


/*     25 */	sxar2
/*     25 */	std,s	%f70,[%fp+1951]
/*     25 */	std,s	%f70,[%fp+1967]

.L4830:

/*    112 */	sxar1
/*    112 */	cmp	%xg11,%g0

/*    112 */	ble	.L4834
	nop


.L4831:


/*    112 */	sxar2
/*    112 */	sra	%xg9,%g0,%xg9
/*    ??? */	ldd,s	[%fp+1871],%f52


/*    112 */	sxar2
/*    112 */	sub	%xg11,4,%xg10
/*    112 */	add	%xg9,%xg9,%xg12


/*    112 */	sxar2
/*    112 */	cmp	%xg10,%g0
/*    112 */	add	%xg12,%xg9,%xg12


/*    112 */	sxar2
/*    112 */	sllx	%xg12,5,%xg12
/*    112 */	fmovd,s	%f52,%f48


/*    112 */	sxar2
/*    112 */	fmovd,s	%f48,%f44
/*    112 */	fmovd,s	%f48,%f40


/*    112 */	sxar2
/*    112 */	fmovd,s	%f48,%f36
/*    112 */	fmovd,s	%f48,%f32

/*    112 */	bl	.L4840
	nop


.L4843:


/*    120 */	sxar2
/*    120 */	fzero,s	%f56
/*    120 */	add	%l2,%xg12,%xg13


/*    112 */	sxar2
/*    112 */	fmovd,s	%f48,%f44
/*    112 */	cmp	%xg10,16


/*    120 */	sxar2
/*    120 */	add	%i1,%xg13,%xg14
/*    120 */	add	%i2,%xg13,%xg15


/*    120 */	sxar2
/*    120 */	fmovd,s	%f48,%f40
/*    120 */	fmovd,s	%f48,%f36


/*    120 */	sxar2
/*    120 */	add	%i3,%xg13,%xg16
/*    120 */	fmovd,s	%f48,%f32


/*    120 */	sxar2
/*    120 */	fmovd,s	%f56,%f64
/*    120 */	fmovd,s	%f56,%f60


/*    120 */	sxar2
/*    120 */	fmovd,s	%f64,%f68
/*    120 */	fmovd,s	%f64,%f72


/*    120 */	sxar2
/*    120 */	fmovd,s	%f64,%f76
/*    120 */	fmovd,s	%f64,%f80


/*    120 */	sxar2
/*    120 */	fmovd,s	%f64,%f84
/*    120 */	fmovd,s	%f64,%f88


/*    120 */	sxar2
/*    120 */	fmovd,s	%f64,%f92
/*    120 */	fmovd,s	%f64,%f96


/*    120 */	sxar2
/*    120 */	fmovd,s	%f64,%f100
/*    120 */	fmovd,s	%f64,%f104


/*    120 */	sxar2
/*    120 */	fmovd,s	%f64,%f108
/*    120 */	fmovd,s	%f64,%f112


/*    120 */	sxar2
/*    120 */	fmovd,s	%f64,%f116
/*    120 */	fmovd,s	%f64,%f120

/*    120 */	sxar1
/*    120 */	fmovd,s	%f64,%f124

/*    112 */	bl	.L4979
	nop


.L4975:


.L4982:


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg13],%f34
/*    112 */	ldd,s	[%xg13+32],%f42


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg13+16],%f38
/*    112 */	ldd,s	[%xg13+64],%f50


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg14+-4032],%f58
/*    112 */	ldd,s	[%xg13+48],%f46

/*    112 */	sxar1
/*    112 */	ldd,s	[%xg13+80],%f54

.L4832:


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg14+-4016],%f128
/*    112 */	faddd,s	%f32,%f34,%f32


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg14+-4000],%f130
/*    112 */	faddd,s	%f36,%f38,%f36


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg14+-3984],%f132
/*    112 */	faddd,s	%f40,%f42,%f40


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg14+-3968],%f134
/*    112 */	faddd,s	%f44,%f46,%f44


/*    112 */	sxar2
/*    112 */	faddd,s	%f48,%f50,%f48
/*    112 */	ldd,s	[%xg14+-3952],%f136


/*    112 */	sxar2
/*    112 */	faddd,s	%f52,%f54,%f52
/*    112 */	faddd,s	%f56,%f58,%f56


/*    112 */	sxar2
/*    112 */	faddd,s	%f60,%f128,%f60
/*    112 */	faddd,s	%f64,%f130,%f64


/*    112 */	sxar2
/*    112 */	faddd,s	%f68,%f132,%f68
/*    112 */	faddd,s	%f72,%f134,%f72


/*    112 */	sxar2
/*    112 */	faddd,s	%f76,%f136,%f76
/*    112 */	ldd,s	[%xg15+-4032],%f138


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg15+-4016],%f140
/*    112 */	ldd,s	[%xg15+-4000],%f142


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg15+-3984],%f144
/*    112 */	ldd,s	[%xg15+-3968],%f146


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg15+-3952],%f148
/*    112 */	ldd,s	[%xg16+-4032],%f150


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg16+-4016],%f152
/*    112 */	ldd,s	[%xg16+-4000],%f154


/*    112 */	sxar2
/*    112 */	faddd,s	%f80,%f138,%f80
/*    112 */	ldd,s	[%xg16+-3984],%f156


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg16+-3968],%f158
/*    112 */	faddd,s	%f84,%f140,%f84


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg16+-3952],%f160
/*    112 */	faddd,s	%f88,%f142,%f88


/*    112 */	sxar2
/*    112 */	faddd,s	%f92,%f144,%f92
/*    112 */	faddd,s	%f96,%f146,%f96


/*    112 */	sxar2
/*    112 */	faddd,s	%f100,%f148,%f100
/*    112 */	faddd,s	%f104,%f150,%f104


/*    112 */	sxar2
/*    112 */	faddd,s	%f108,%f152,%f108
/*    112 */	faddd,s	%f112,%f154,%f112


/*    112 */	sxar2
/*    112 */	faddd,s	%f116,%f156,%f116
/*    112 */	add	%l0,%xg13,%xg13


/*    112 */	sxar2
/*    112 */	faddd,s	%f120,%f158,%f120
/*    112 */	faddd,s	%f124,%f160,%f124


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg13],%f162
/*    112 */	ldd,s	[%xg13+32],%f166


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg13+16],%f164
/*    112 */	add	%l0,%xg14,%xg14


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg13+64],%f170
/*    112 */	ldd,s	[%xg14+-4032],%f174


/*    112 */	sxar2
/*    112 */	add	%xg12,%l0,%xg12
/*    112 */	ldd,s	[%xg13+48],%f168


/*    112 */	sxar2
/*    112 */	add	%l0,%xg16,%xg16
/*    112 */	add	%l0,%xg15,%xg15


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg13+80],%f172
/*    112 */	ldd,s	[%xg14+-4016],%f176


/*    112 */	sxar2
/*    112 */	faddd,s	%f32,%f162,%f32
/*    112 */	ldd,s	[%xg14+-4000],%f178


/*    112 */	sxar2
/*    112 */	faddd,s	%f36,%f164,%f36
/*    112 */	ldd,s	[%xg14+-3984],%f180


/*    112 */	sxar2
/*    112 */	faddd,s	%f40,%f166,%f40
/*    112 */	ldd,s	[%xg14+-3968],%f182


/*    112 */	sxar2
/*    112 */	faddd,s	%f44,%f168,%f44
/*    112 */	faddd,s	%f48,%f170,%f48


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg14+-3952],%f184
/*    112 */	faddd,s	%f52,%f172,%f52


/*    112 */	sxar2
/*    112 */	faddd,s	%f56,%f174,%f56
/*    112 */	faddd,s	%f60,%f176,%f60


/*    112 */	sxar2
/*    112 */	faddd,s	%f64,%f178,%f64
/*    112 */	faddd,s	%f68,%f180,%f68


/*    112 */	sxar2
/*    112 */	faddd,s	%f72,%f182,%f72
/*    112 */	faddd,s	%f76,%f184,%f76


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg15+-4032],%f186
/*    112 */	ldd,s	[%xg15+-4016],%f188


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg15+-4000],%f190
/*    112 */	ldd,s	[%xg15+-3984],%f192


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg15+-3968],%f194
/*    112 */	ldd,s	[%xg15+-3952],%f196


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg16+-4032],%f198
/*    112 */	ldd,s	[%xg16+-4016],%f200


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg16+-4000],%f202
/*    112 */	faddd,s	%f80,%f186,%f80


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg16+-3984],%f204
/*    112 */	ldd,s	[%xg16+-3968],%f206


/*    112 */	sxar2
/*    112 */	faddd,s	%f84,%f188,%f84
/*    112 */	ldd,s	[%xg16+-3952],%f208


/*    112 */	sxar2
/*    112 */	faddd,s	%f88,%f190,%f88
/*    112 */	faddd,s	%f92,%f192,%f92


/*    112 */	sxar2
/*    112 */	faddd,s	%f96,%f194,%f96
/*    112 */	faddd,s	%f100,%f196,%f100


/*    112 */	sxar2
/*    112 */	faddd,s	%f104,%f198,%f104
/*    112 */	faddd,s	%f108,%f200,%f108


/*    112 */	sxar2
/*    112 */	faddd,s	%f112,%f202,%f112
/*    112 */	faddd,s	%f116,%f204,%f116


/*    112 */	sxar2
/*    112 */	add	%l0,%xg13,%xg13
/*    112 */	faddd,s	%f120,%f206,%f120


/*    112 */	sxar2
/*    112 */	faddd,s	%f124,%f208,%f124
/*    112 */	ldd,s	[%xg13],%f34


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg13+32],%f42
/*    112 */	ldd,s	[%xg13+16],%f38


/*    112 */	sxar2
/*    112 */	add	%l0,%xg14,%xg14
/*    112 */	ldd,s	[%xg13+64],%f50


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg14+-4032],%f58
/*    112 */	add	%xg12,%l0,%xg12


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg13+48],%f46
/*    112 */	add	%l0,%xg16,%xg16


/*    112 */	sxar2
/*    112 */	add	%l0,%xg15,%xg15
/*    112 */	ldd,s	[%xg13+80],%f54


/*    112 */	sxar2
/*    112 */	sub	%xg10,8,%xg10
/*    112 */	cmp	%xg10,19

/*    112 */	bge,pt	%icc, .L4832
	nop


.L4983:


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg14+-4016],%f62
/*    112 */	faddd,s	%f32,%f34,%f32


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg14+-4000],%f66
/*    112 */	faddd,s	%f36,%f38,%f36


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg14+-3984],%f70
/*    112 */	faddd,s	%f40,%f42,%f40


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg14+-3968],%f74
/*    112 */	faddd,s	%f44,%f46,%f44


/*    112 */	sxar2
/*    112 */	faddd,s	%f48,%f50,%f48
/*    112 */	ldd,s	[%xg14+-3952],%f78


/*    112 */	sxar2
/*    112 */	faddd,s	%f52,%f54,%f52
/*    112 */	ldd,s	[%xg15+-4032],%f82


/*    112 */	sxar2
/*    112 */	faddd,s	%f56,%f58,%f56
/*    112 */	ldd,s	[%xg15+-4016],%f86


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg15+-4000],%f90
/*    112 */	add	%l0,%xg13,%xg13


/*    112 */	sxar2
/*    112 */	faddd,s	%f60,%f62,%f60
/*    112 */	faddd,s	%f64,%f66,%f64


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg15+-3984],%f94
/*    112 */	ldd,s	[%xg15+-3968],%f98


/*    112 */	sxar2
/*    112 */	faddd,s	%f68,%f70,%f68
/*    112 */	faddd,s	%f72,%f74,%f72


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg15+-3952],%f102
/*    112 */	ldd,s	[%xg16+-4032],%f106


/*    112 */	sxar2
/*    112 */	faddd,s	%f76,%f78,%f76
/*    112 */	ldd,s	[%xg16+-4016],%f110


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg16+-4000],%f114
/*    112 */	faddd,s	%f80,%f82,%f80


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg16+-3984],%f118
/*    112 */	ldd,s	[%xg16+-3968],%f122


/*    112 */	sxar2
/*    112 */	faddd,s	%f84,%f86,%f84
/*    112 */	faddd,s	%f88,%f90,%f88


/*    112 */	sxar2
/*    112 */	ldd,s	[%xg16+-3952],%f126
/*    112 */	faddd,s	%f92,%f94,%f92


/*    112 */	sxar2
/*    112 */	faddd,s	%f96,%f98,%f96
/*    112 */	add	%l0,%xg14,%xg14


/*    112 */	sxar2
/*    112 */	faddd,s	%f100,%f102,%f100
/*    112 */	faddd,s	%f104,%f106,%f104


/*    112 */	sxar2
/*    112 */	add	%xg12,%l0,%xg12
/*    112 */	add	%l0,%xg16,%xg16


/*    112 */	sxar2
/*    112 */	faddd,s	%f108,%f110,%f108
/*    112 */	faddd,s	%f112,%f114,%f112


/*    112 */	sxar2
/*    112 */	add	%l0,%xg15,%xg15
/*    112 */	sub	%xg10,4,%xg10


/*    112 */	sxar2
/*    112 */	faddd,s	%f116,%f118,%f116
/*    112 */	faddd,s	%f120,%f122,%f120

/*    112 */	sxar1
/*    112 */	faddd,s	%f124,%f126,%f124

.L4979:


.L4978:


.L4981:


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg13],%f210
/*     43 */	ldd,s	[%xg13+16],%f212


/*    120 */	sxar2
/*    120 */	add	%xg12,%l0,%xg12
/*    120 */	subcc	%xg10,4,%xg10


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg13+32],%f214
/*     43 */	ldd,s	[%xg13+48],%f216


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg13+64],%f218
/*     43 */	ldd,s	[%xg13+80],%f220


/*     43 */	sxar2
/*     43 */	add	%l0,%xg13,%xg13
/*     43 */	ldd,s	[%xg14+-4032],%f222


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg14+-4016],%f224
/*     43 */	faddd,s	%f32,%f210,%f32


/*     43 */	sxar2
/*     43 */	faddd,s	%f36,%f212,%f36
/*     43 */	ldd,s	[%xg14+-4000],%f226


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg14+-3984],%f228
/*     43 */	faddd,s	%f40,%f214,%f40


/*     43 */	sxar2
/*     43 */	faddd,s	%f44,%f216,%f44
/*     43 */	ldd,s	[%xg14+-3968],%f230


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg14+-3952],%f232
/*     43 */	faddd,s	%f48,%f218,%f48


/*     43 */	sxar2
/*     43 */	faddd,s	%f52,%f220,%f52
/*     43 */	ldd,s	[%xg15+-4032],%f234


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg15+-4016],%f236
/*     43 */	faddd,s	%f56,%f222,%f56


/*     43 */	sxar2
/*     43 */	faddd,s	%f60,%f224,%f60
/*     43 */	ldd,s	[%xg15+-4000],%f238


/*     43 */	sxar2
/*     43 */	add	%l0,%xg14,%xg14
/*     43 */	faddd,s	%f64,%f226,%f64


/*     43 */	sxar2
/*     43 */	faddd,s	%f68,%f228,%f68
/*     43 */	faddd,s	%f72,%f230,%f72


/*     43 */	sxar2
/*     43 */	faddd,s	%f76,%f232,%f76
/*     43 */	faddd,s	%f80,%f234,%f80


/*     43 */	sxar2
/*     43 */	faddd,s	%f84,%f236,%f84
/*     43 */	ldd,s	[%xg15+-3984],%f240


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg15+-3968],%f242
/*     43 */	faddd,s	%f88,%f238,%f88


/*    120 */	sxar2
/*    120 */	ldd,s	[%xg15+-3952],%f244
/*    120 */	add	%l0,%xg15,%xg15


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg16+-4032],%f246
/*     43 */	ldd,s	[%xg16+-4016],%f248


/*     43 */	sxar2
/*     43 */	faddd,s	%f92,%f240,%f92
/*     43 */	faddd,s	%f96,%f242,%f96


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg16+-4000],%f250
/*     43 */	ldd,s	[%xg16+-3984],%f252


/*     43 */	sxar2
/*     43 */	faddd,s	%f100,%f244,%f100
/*     43 */	ldd,s	[%xg16+-3968],%f254


/*    120 */	sxar2
/*    120 */	ldd,s	[%xg16+-3952],%f34
/*    120 */	add	%l0,%xg16,%xg16


/*     43 */	sxar2
/*     43 */	faddd,s	%f104,%f246,%f104
/*     43 */	faddd,s	%f108,%f248,%f108


/*     43 */	sxar2
/*     43 */	faddd,s	%f112,%f250,%f112
/*     43 */	faddd,s	%f116,%f252,%f116


/*     43 */	sxar2
/*     43 */	faddd,s	%f120,%f254,%f120
/*     43 */	faddd,s	%f124,%f34,%f124

/*    120 */	bpos,pt	%icc, .L4981
	nop


.L4977:


/*    120 */	sxar2
/*    120 */	faddd,s	%f100,%f124,%f100
/*    120 */	faddd,s	%f52,%f76,%f52


/*    120 */	sxar2
/*    120 */	faddd,s	%f96,%f120,%f96
/*    120 */	faddd,s	%f48,%f72,%f48


/*    120 */	sxar2
/*    120 */	faddd,s	%f92,%f116,%f92
/*    120 */	faddd,s	%f44,%f68,%f44


/*    120 */	sxar2
/*    120 */	faddd,s	%f88,%f112,%f88
/*    120 */	faddd,s	%f40,%f64,%f40


/*    120 */	sxar2
/*    120 */	faddd,s	%f84,%f108,%f84
/*    120 */	faddd,s	%f36,%f60,%f36


/*    120 */	sxar2
/*    120 */	faddd,s	%f80,%f104,%f80
/*    120 */	faddd,s	%f32,%f56,%f32


/*    120 */	sxar2
/*    120 */	faddd,s	%f52,%f100,%f52
/*    120 */	faddd,s	%f48,%f96,%f48


/*    120 */	sxar2
/*    120 */	faddd,s	%f44,%f92,%f44
/*    120 */	faddd,s	%f40,%f88,%f40


/*    120 */	sxar2
/*    120 */	faddd,s	%f36,%f84,%f36
/*    120 */	faddd,s	%f32,%f80,%f32

.L4840:

/*    112 */	sxar1
/*    112 */	addcc	%xg10,3,%xg10

/*    112 */	bneg	.L4833
	nop


.L4841:

/*    112 */	sxar1
/*    112 */	add	%l2,%xg12,%xg12

.L4848:


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg12],%f38
/*     43 */	ldd,s	[%xg12+16],%f42


/*     43 */	sxar2
/*     43 */	subcc	%xg10,1,%xg10
/*     43 */	ldd,s	[%xg12+32],%f46


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg12+48],%f50
/*     43 */	ldd,s	[%xg12+64],%f54


/*    120 */	sxar2
/*    120 */	ldd,s	[%xg12+80],%f56
/*    120 */	add	%i4,%xg12,%xg12


/*     43 */	sxar2
/*     43 */	faddd,s	%f32,%f38,%f32
/*     43 */	faddd,s	%f36,%f42,%f36


/*     43 */	sxar2
/*     43 */	faddd,s	%f40,%f46,%f40
/*     43 */	faddd,s	%f44,%f50,%f44


/*     43 */	sxar2
/*     43 */	faddd,s	%f48,%f54,%f48
/*     43 */	faddd,s	%f52,%f56,%f52

/*    120 */	bpos,pt	%icc, .L4848
	nop


.L4842:


.L4833:


/*    120 */	sxar2
/*    120 */	std,s	%f32,[%fp+1887]
/*    120 */	std,s	%f36,[%fp+1903]


/*    120 */	sxar2
/*    120 */	std,s	%f40,[%fp+1919]
/*    120 */	std,s	%f44,[%fp+1935]


/*    120 */	sxar2
/*    120 */	std,s	%f48,[%fp+1951]
/*    120 */	std,s	%f52,[%fp+1967]

.L4834:



/*    134 */	sxar2
/*    134 */	ldd,s	[%fp+1887],%f58
/*    134 */	add	%xg7,2,%xg7



/*     81 */	sxar2
/*     81 */	subcc	%xg0,1,%xg0
/*     81 */	std	%f58,[%xg4]



/*     81 */	sxar2
/*     81 */	ldd,s	[%fp+1903],%f60
/*     81 */	std	%f60,[%xg4+8]



/*     81 */	sxar2
/*     81 */	ldd,s	[%fp+1919],%f62
/*     81 */	std	%f62,[%xg4+16]



/*     81 */	sxar2
/*     81 */	ldd,s	[%fp+1935],%f64
/*     81 */	std	%f64,[%xg4+24]



/*     81 */	sxar2
/*     81 */	ldd,s	[%fp+1951],%f66
/*     81 */	std	%f66,[%xg4+32]



/*     81 */	sxar2
/*     81 */	ldd,s	[%fp+1967],%f68
/*     81 */	std	%f68,[%xg4+40]


/*     84 */	sxar2
/*     84 */	add	%xg4,96,%xg4
/*     84 */	std	%f314,[%xg8]


/*     84 */	sxar2
/*     84 */	std	%f316,[%xg8+8]
/*     84 */	std	%f318,[%xg8+16]


/*     84 */	sxar2
/*     84 */	std	%f320,[%xg8+24]
/*     84 */	std	%f322,[%xg8+32]


/*    134 */	sxar2
/*    134 */	std	%f324,[%xg8+40]
/*    134 */	add	%xg8,96,%xg8

/*    134 */	bne,pt	%icc, .L4829
/*    134 */	add	%o0,2,%o0


.L4835:

/*    134 */
/*    134 */	ba	.L4827
	nop


.L4837:

/*    134 *//*    134 */	call	__mpc_obar
/*    134 */	ldx	[%fp+2199],%o0

/*    134 *//*    134 */	call	__mpc_obar
/*    134 */	ldx	[%fp+2199],%o0


.L4838:

/*    134 */	ret
	restore



.LLFE8:
	.size	_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_3,.-_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_3
	.type	_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_3,#function
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite4-k.cpp _ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd $"
	.section	".text"
	.global	_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd
	.align	64
_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd:
.LLFB9:
.L667:

/*    138 */	save	%sp,-880,%sp
.LLCFI7:
/*    138 */	stw	%i0,[%fp+2179]
/*    138 */	std	%f2,[%fp+2183]
/*    138 */	stx	%i2,[%fp+2191]
/*    138 */	stx	%i3,[%fp+2199]
/*    138 */	stx	%i4,[%fp+2207]

.L668:

/*    146 *//*    146 */	sethi	%h44(_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_4),%o0
/*    146 */	mov	%fp,%l0
/*    146 */	or	%o0,%m44(_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_4),%o0
/*    146 */	mov	%g0,%l1
/*    146 */	sllx	%o0,12,%o0
/*    146 */	mov	%l0,%o1
/*    146 */	or	%o0,%l44(_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_4),%o0
/*    146 */	call	__mpc_opar
/*    146 */	mov	%l1,%o2

/*    156 */
/*    158 *//*    158 */	sethi	%h44(_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_5),%o0
/*    158 */	mov	%l0,%o1
/*    158 */	or	%o0,%m44(_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_5),%o0
/*    158 */	mov	%l1,%o2
/*    158 */	sllx	%o0,12,%o0
/*    158 */	call	__mpc_opar
/*    158 */	or	%o0,%l44(_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_5),%o0

/*    180 */
/*    180 */	ret
	restore



.L712:


.LLFE9:
	.size	_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd,.-_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd
	.type	_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd,#function
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite4-k.cpp _ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_4 $"
	.section	".text"
	.align	64
_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_4:
.LLFB10:
.L4850:

/*    146 */	save	%sp,-640,%sp
.LLCFI8:
/*    146 */	stx	%i0,[%fp+2175]
/*    146 */	stx	%i3,[%fp+2199]
/*    146 */	stx	%i0,[%fp+2175]

.L4851:

/*    146 *//*    146 */	sxar1
/*    146 */	ldsw	[%i0+2031],%xg9
/*    146 */
/*    146 */
/*    146 */
/*    147 */	ldsw	[%i0+2179],%l0
/*    147 */	cmp	%l0,%g0
/*    147 */	ble	.L4858
/*    147 */	mov	%g0,%o0


.L4852:

/*    147 */	sxar1
/*    147 */	mov	1,%xg8

/*    147 */	mov	1,%l5

/*    147 */	sxar1
/*    147 */	stx	%xg8,[%fp+2031]

/*    147 */	sra	%l0,%g0,%l0

/*    147 */	add	%fp,2039,%l1

/*    147 */	add	%fp,2023,%l2

/*    147 */	add	%fp,2031,%l3

/*    147 */	sra	%l5,%g0,%l4

.L4853:

/*    147 */	sra	%o0,%g0,%o0

/*    147 */	stx	%g0,[%sp+2223]

/*    147 */	mov	2,%o2

/*    147 */	mov	%g0,%o3

/*    147 */	mov	%l0,%o1

/*    147 */	mov	%l1,%o4


/*    147 */	stx	%g0,[%sp+2231]

/*    147 */	stx	%l3,[%sp+2239]


/*    147 */	sxar2
/*    147 */	ldx	[%fp+2199],%xg6
/*    147 */	stx	%xg6,[%sp+2247]

/*    147 */	call	__mpc_ostd_th
/*    147 */	mov	%l2,%o5
/*    147 */	sxar2
/*    147 */	ldx	[%fp+2031],%xg7
/*    147 */	cmp	%xg7,%g0
/*    147 */	ble,pn	%xcc, .L4858
	nop


.L4854:

/*    147 */	ldx	[%fp+2039],%o0

/*    147 */	sxar1
/*    147 */	ldx	[%fp+2023],%xg0

/*    147 */	sra	%o0,%g0,%o0


/*    147 */	sxar2
/*    147 */	sra	%xg0,%g0,%xg0
/*    147 */	sub	%xg0,%o0,%xg0


/*    147 */	sxar2
/*    147 */	add	%o0,1,%xg1
/*    147 */	srl	%xg0,31,%xg2


/*    147 */	sxar2
/*    147 */	sra	%o0,%g0,%xg3
/*    147 */	add	%xg0,%xg2,%xg0


/*    147 */	sxar2
/*    147 */	sra	%xg1,%g0,%xg1
/*    147 */	sra	%xg0,1,%xg0


/*    147 */	sxar2
/*    147 */	sllx	%xg3,5,%xg3
/*    147 */	add	%xg0,1,%xg0


/*    147 */	sxar2
/*    147 */	sllx	%xg1,5,%xg1
/*    147 */	sra	%xg0,%g0,%xg0


/*    147 */	sxar2
/*    147 */	sub	%l4,%xg0,%xg0
/*    147 */	srax	%xg0,32,%xg4


/*    147 */	sxar2
/*    147 */	and	%xg0,%xg4,%xg0
/*    147 */	sub	%l5,%xg0,%xg0

/*    147 */	sxar1
/*    147 */	subcc	%xg0,4,%xg0

/*    147 */	bneg	.L4861
	nop


.L4864:


/*    147 */	sxar2
/*    147 */	ldx	[%i0+2191],%xg6
/*    147 */	ldx	[%i0+2199],%xg10


/*    147 */	sxar2
/*    147 */	cmp	%xg0,16
/*    147 */	add	%xg6,16,%xg5


/*    147 */	sxar2
/*    147 */	add	%xg10,%xg3,%xg9
/*    147 */	add	%xg6,32,%xg7


/*    147 */	sxar2
/*    147 */	add	%xg6,48,%xg8
/*    147 */	add	%xg10,%xg1,%xg10

/*    147 */	bl	.L4988
	nop


.L4984:


.L4991:


/*    147 */	sxar2
/*    147 */	srl	%o0,31,%xg11
/*    147 */	add	%o0,2,%xg12


/*    147 */	sxar2
/*    147 */	add	%xg11,%o0,%xg11
/*    147 */	srl	%xg12,31,%xg13


/*    147 */	sxar2
/*    147 */	sra	%xg11,1,%xg11
/*    147 */	add	%xg12,%xg13,%xg13


/*    147 */	sxar2
/*    147 */	sra	%xg11,%g0,%xg11
/*    147 */	sra	%xg13,1,%xg13


/*    147 */	sxar2
/*    147 */	sllx	%xg11,3,%xg14
/*    147 */	sub	%xg14,%xg11,%xg14


/*    147 */	sxar2
/*    147 */	sllx	%xg14,5,%xg14
/*    147 */	add	%xg14,%xg6,%xg15

.L4855:


/*    147 */	sxar2
/*    147 */	add	%xg14,%xg5,%o2
/*    147 */	sra	%xg13,%g0,%xg13


/*    147 */	sxar2
/*    147 */	ldd,s	[%xg15],%f80
/*    147 */	add	%xg14,%xg7,%o3


/*    147 */	sxar2
/*    147 */	ldd,s	[%o2],%f84
/*    147 */	sllx	%xg13,3,%o4


/*    147 */	sxar2
/*    147 */	add	%xg14,%xg8,%xg14
/*    147 */	sub	%o4,%xg13,%o4


/*    147 */	sxar2
/*    147 */	ldd,s	[%o3],%f86
/*    147 */	ldd,s	[%xg14],%f90

/*    147 */	sllx	%o4,5,%o4


/*    147 */	sxar2
/*    147 */	add	%xg12,2,%o5
/*    147 */	add	%o4,%xg6,%o7


/*    147 */	sxar2
/*    147 */	srl	%o5,31,%xg2
/*    147 */	add	%o4,%xg5,%xg4



/*    147 */	sxar2
/*    147 */	add	%o5,%xg2,%o5
/*    147 */	fmovd	%f336,%f82



/*    147 */	sxar2
/*    147 */	fmovd	%f340,%f338
/*    147 */	fmovd	%f84,%f336


/*    147 */	sxar2
/*    147 */	ldd,s	[%o7],%f92
/*    147 */	add	%o4,%xg7,%xg11

/*    147 */	sxar1
/*    147 */	ldd,s	[%xg4],%f96

/*    147 */	sra	%o5,1,%o5


/*    147 */	sxar1
/*    147 */	add	%o4,%xg8,%o4

/*    147 */	sra	%o5,%g0,%o5




/*    147 */	sxar2
/*    147 */	fmovd	%f342,%f88
/*    147 */	fmovd	%f346,%f344


/*    147 */	sxar2
/*    147 */	fmovd	%f90,%f342
/*    147 */	std,s	%f80,[%xg9]


/*    147 */	sxar2
/*    147 */	sllx	%o5,3,%xg13
/*    147 */	add	%xg12,4,%xg14



/*    147 */	sxar2
/*    147 */	ldd,s	[%xg11],%f98
/*    147 */	ldd,s	[%o4],%f102


/*    147 */	sxar2
/*    147 */	sub	%xg13,%o5,%xg13
/*    147 */	srl	%xg14,31,%xg15



/*    147 */	sxar2
/*    147 */	std,s	%f86,[%xg9+16]
/*    147 */	sllx	%xg13,5,%xg13




/*    147 */	sxar2
/*    147 */	add	%xg14,%xg15,%xg14
/*    147 */	fmovd	%f348,%f94



/*    147 */	sxar2
/*    147 */	fmovd	%f352,%f350
/*    147 */	fmovd	%f96,%f348


/*    147 */	sxar2
/*    147 */	std,s	%f82,[%xg10]
/*    147 */	add	%xg13,%xg6,%xg16



/*    147 */	sxar2
/*    147 */	sra	%xg14,1,%xg14
/*    147 */	std,s	%f88,[%xg10+16]


/*    147 */	sxar2
/*    147 */	add	%xg13,%xg5,%xg17
/*    147 */	sra	%xg14,%g0,%xg14


/*    147 */	sxar2
/*    147 */	std,s	%f92,[%xg9+64]
/*    147 */	add	%xg13,%xg7,%xg18




/*    147 */	sxar2
/*    147 */	sllx	%xg14,3,%xg19
/*    147 */	ldd,s	[%xg16],%f104


/*    147 */	sxar2
/*    147 */	add	%xg13,%xg8,%xg13
/*    147 */	ldd,s	[%xg17],%f108


/*    147 */	sxar2
/*    147 */	sub	%xg19,%xg14,%xg19
/*    147 */	fmovd	%f98,%f100



/*    147 */	sxar2
/*    147 */	fmovd	%f102,%f356
/*    147 */	ldd,s	[%xg18],%f110


/*    147 */	sxar2
/*    147 */	add	%xg12,6,%xg20
/*    147 */	ldd,s	[%xg13],%f114



/*    147 */	sxar2
/*    147 */	sllx	%xg19,5,%xg19
/*    147 */	std,s	%f100,[%xg9+80]


/*    147 */	sxar2
/*    147 */	srl	%xg20,31,%xg21
/*    147 */	add	%xg19,%xg6,%xg22


/*    147 */	sxar2
/*    147 */	std,s	%f94,[%xg10+64]
/*    147 */	add	%xg19,%xg5,%xg23


/*    147 */	sxar2
/*    147 */	add	%xg19,%xg7,%xg24
/*    147 */	ldd,s	[%xg22],%f116


/*    147 */	sxar2
/*    147 */	add	%xg19,%xg8,%xg19
/*    147 */	ldd,s	[%xg23],%f120




/*    147 */	sxar2
/*    147 */	fmovd	%f354,%f102
/*    147 */	fmovd	%f360,%f106



/*    147 */	sxar2
/*    147 */	fmovd	%f364,%f362
/*    147 */	ldd,s	[%xg24],%f122


/*    147 */	sxar2
/*    147 */	ldd,s	[%xg19],%f126
/*    147 */	fmovd	%f108,%f360





/*    147 */	sxar2
/*    147 */	fmovd	%f366,%f112
/*    147 */	fmovd	%f370,%f368


/*    147 */	sxar2
/*    147 */	fmovd	%f114,%f366
/*    147 */	std,s	%f102,[%xg10+80]



/*    147 */	sxar2
/*    147 */	add	%xg21,%xg20,%xg21
/*    147 */	std,s	%f104,[%xg9+128]



/*    147 */	sxar2
/*    147 */	sra	%xg21,1,%xg21
/*    147 */	std,s	%f110,[%xg9+144]


/*    147 */	sxar2
/*    147 */	sra	%xg21,%g0,%xg21
/*    147 */	add	%xg12,8,%xg25





/*    147 */	sxar2
/*    147 */	std,s	%f106,[%xg10+128]
/*    147 */	sllx	%xg21,3,%xg26


/*    147 */	sxar2
/*    147 */	srl	%xg25,31,%xg27
/*    147 */	fmovd	%f116,%f118





/*    147 */	sxar2
/*    147 */	fmovd	%f120,%f374
/*    147 */	std,s	%f112,[%xg10+144]


/*    147 */	sxar2
/*    147 */	sub	%xg26,%xg21,%xg26
/*    147 */	add	%xg25,%xg27,%xg25


/*    147 */	sxar2
/*    147 */	fmovd	%f122,%f124
/*    147 */	fmovd	%f126,%f380



/*    147 */	sxar2
/*    147 */	std,s	%f118,[%xg9+192]
/*    147 */	sllx	%xg26,5,%xg26


/*    147 */	sxar2
/*    147 */	sra	%xg25,1,%xg25
/*    147 */	fmovd	%f372,%f120



/*    147 */	sxar2
/*    147 */	std,s	%f124,[%xg9+208]
/*    147 */	add	%xg26,%xg6,%xg28



/*    147 */	sxar2
/*    147 */	fmovd	%f378,%f126
/*    147 */	std,s	%f120,[%xg10+192]


/*    147 */	sxar2
/*    147 */	std,s	%f126,[%xg10+208]
/*    147 */	add	%xg26,%xg5,%xg29


/*    147 */	sxar2
/*    147 */	sra	%xg25,%g0,%xg25
/*    147 */	ldd,s	[%xg28],%f128


/*    147 */	sxar2
/*    147 */	add	%xg26,%xg7,%xg30
/*    147 */	ldd,s	[%xg29],%f132


/*    147 */	sxar2
/*    147 */	sllx	%xg25,3,%xg31
/*    147 */	add	%xg26,%xg8,%xg26


/*    147 */	sxar2
/*    147 */	sub	%xg31,%xg25,%xg31
/*    147 */	ldd,s	[%xg30],%f134


/*    147 */	sxar2
/*    147 */	ldd,s	[%xg26],%f138
/*    147 */	sllx	%xg31,5,%xg31


/*    147 */	sxar2
/*    147 */	add	%xg12,10,%g1
/*    147 */	add	%xg31,%xg6,%g2

/*    147 */	srl	%g1,31,%g3

/*    147 */	sxar1
/*    147 */	add	%xg31,%xg5,%g4

/*    147 */	add	%g1,%g3,%g1




/*    147 */	sxar2
/*    147 */	ldd,s	[%g2],%f140
/*    147 */	add	%xg31,%xg7,%g5

/*    147 */	sxar1
/*    147 */	ldd,s	[%g4],%f144

/*    147 */	sra	%g1,1,%g1


/*    147 */	sxar2
/*    147 */	fmovd	%f128,%f130
/*    147 */	fmovd	%f132,%f386


/*    147 */	sxar1
/*    147 */	add	%xg31,%xg8,%xg31

/*    147 */	sra	%g1,%g0,%g1



/*    147 */	sxar1
/*    147 */	std,s	%f130,[%xg9+256]

/*    147 */	sllx	%g1,3,%o0


/*    147 */	sxar2
/*    147 */	add	%xg12,12,%o1
/*    147 */	fmovd	%f134,%f136



/*    147 */	sxar2
/*    147 */	fmovd	%f138,%f392
/*    147 */	ldd,s	[%g5],%f146

/*    147 */	sxar1
/*    147 */	ldd,s	[%xg31],%f150

/*    147 */	sub	%o0,%g1,%o0

/*    147 */	srl	%o1,31,%o2



/*    147 */	sxar2
/*    147 */	fmovd	%f384,%f132
/*    147 */	std,s	%f136,[%xg9+272]

/*    147 */	sllx	%o0,5,%o0

/*    147 */	add	%o1,%o2,%o1




/*    147 */	sxar2
/*    147 */	fmovd	%f390,%f138
/*    147 */	fmovd	%f396,%f142



/*    147 */	sxar2
/*    147 */	fmovd	%f400,%f398
/*    147 */	fmovd	%f144,%f396


/*    147 */	sxar2
/*    147 */	std,s	%f132,[%xg10+256]
/*    147 */	add	%o0,%xg6,%o3

/*    147 */	sra	%o1,1,%o1



/*    147 */	sxar2
/*    147 */	std,s	%f138,[%xg10+272]
/*    147 */	add	%o0,%xg5,%o4

/*    147 */	sra	%o1,%g0,%o1


/*    147 */	sxar2
/*    147 */	std,s	%f140,[%xg9+320]
/*    147 */	add	%o0,%xg7,%o5

/*    147 */	sllx	%o1,3,%o7




/*    147 */	sxar2
/*    147 */	ldd,s	[%o3],%f152
/*    147 */	add	%o0,%xg8,%o0

/*    147 */	sxar1
/*    147 */	ldd,s	[%o4],%f156

/*    147 */	sub	%o7,%o1,%o7


/*    147 */	sxar2
/*    147 */	fmovd	%f146,%f148
/*    147 */	fmovd	%f150,%f404



/*    147 */	sxar2
/*    147 */	ldd,s	[%o5],%f158
/*    147 */	add	%xg12,14,%xg2

/*    147 */	sxar1
/*    147 */	ldd,s	[%o0],%f162

/*    147 */	sllx	%o7,5,%o7



/*    147 */	sxar2
/*    147 */	std,s	%f148,[%xg9+336]
/*    147 */	srl	%xg2,31,%xg4


/*    147 */	sxar2
/*    147 */	add	%o7,%xg6,%xg11
/*    147 */	std,s	%f142,[%xg10+320]


/*    147 */	sxar2
/*    147 */	add	%o7,%xg5,%xg13
/*    147 */	add	%o7,%xg7,%xg14


/*    147 */	sxar2
/*    147 */	ldd,s	[%xg11],%f164
/*    147 */	add	%o7,%xg8,%o7


/*    147 */	sxar2
/*    147 */	ldd,s	[%xg13],%f168
/*    147 */	fmovd	%f402,%f150





/*    147 */	sxar2
/*    147 */	ldd,s	[%xg14],%f170
/*    147 */	ldd,s	[%o7],%f174


/*    147 */	sxar2
/*    147 */	fmovd	%f152,%f154
/*    147 */	fmovd	%f156,%f410





/*    147 */	sxar2
/*    147 */	std,s	%f150,[%xg10+336]
/*    147 */	add	%xg4,%xg2,%xg4


/*    147 */	sxar2
/*    147 */	fmovd	%f158,%f160
/*    147 */	fmovd	%f162,%f416



/*    147 */	sxar2
/*    147 */	std,s	%f154,[%xg9+384]
/*    147 */	sra	%xg4,1,%xg4



/*    147 */	sxar2
/*    147 */	fmovd	%f408,%f156
/*    147 */	std,s	%f160,[%xg9+400]


/*    147 */	sxar2
/*    147 */	sra	%xg4,%g0,%xg4
/*    147 */	add	%xg12,16,%xg12





/*    147 */	sxar2
/*    147 */	fmovd	%f414,%f162
/*    147 */	std,s	%f156,[%xg10+384]


/*    147 */	sxar2
/*    147 */	sllx	%xg4,3,%xg14
/*    147 */	srl	%xg12,31,%xg13


/*    147 */	sxar2
/*    147 */	fmovd	%f164,%f166
/*    147 */	fmovd	%f168,%f422





/*    147 */	sxar2
/*    147 */	fmovd	%f426,%f172
/*    147 */	fmovd	%f430,%f428


/*    147 */	sxar2
/*    147 */	fmovd	%f174,%f426
/*    147 */	std,s	%f162,[%xg10+400]


/*    147 */	sxar2
/*    147 */	sub	%xg14,%xg4,%xg14
/*    147 */	add	%xg12,%xg13,%xg13



/*    147 */	sxar2
/*    147 */	std,s	%f166,[%xg9+448]
/*    147 */	sllx	%xg14,5,%xg14


/*    147 */	sxar2
/*    147 */	sra	%xg13,1,%xg13
/*    147 */	fmovd	%f420,%f168



/*    147 */	sxar2
/*    147 */	std,s	%f170,[%xg9+464]
/*    147 */	add	%xg14,%xg6,%xg15



/*    147 */	sxar2
/*    147 */	add	%xg3,512,%xg3
/*    147 */	std,s	%f168,[%xg10+448]


/*    147 */	sxar2
/*    147 */	add	%xg9,512,%xg9
/*    147 */	add	%xg1,512,%xg1


/*    147 */	sxar2
/*    147 */	std,s	%f172,[%xg10+464]
/*    147 */	add	%xg10,512,%xg10


/*    147 */	sxar2
/*    147 */	sub	%xg0,8,%xg0
/*    147 */	cmp	%xg0,19

/*    147 */	bge,pt	%icc, .L4855
	nop


.L4992:


/*    147 */	sxar2
/*    147 */	add	%xg14,%xg5,%xg16
/*    147 */	ldd,s	[%xg15],%f32


/*    147 */	sxar2
/*    147 */	add	%xg14,%xg7,%xg17
/*    147 */	sra	%xg13,%g0,%xg13


/*    147 */	sxar2
/*    147 */	ldd,s	[%xg16],%f36
/*    147 */	add	%xg14,%xg8,%xg14


/*    147 */	sxar2
/*    147 */	ldd,s	[%xg17],%f38
/*    147 */	sllx	%xg13,3,%xg18


/*    147 */	sxar2
/*    147 */	ldd,s	[%xg14],%f42
/*    147 */	add	%xg12,2,%xg19


/*    147 */	sxar2
/*    147 */	sub	%xg18,%xg13,%xg18
/*    147 */	srl	%xg19,31,%xg20


/*    147 */	sxar2
/*    147 */	sllx	%xg18,5,%xg18
/*    147 */	add	%xg19,%xg20,%xg19


/*    147 */	fmovd	%f32,%f34


/*    147 */	sxar2
/*    147 */	add	%xg18,%xg6,%xg21
/*    147 */	add	%xg18,%xg5,%xg22





/*    147 */	sxar2
/*    147 */	fmovd	%f36,%f290
/*    147 */	fmovd	%f294,%f40


/*    147 */	sxar2
/*    147 */	add	%xg18,%xg7,%xg23
/*    147 */	add	%xg18,%xg8,%xg18



/*    147 */	sxar2
/*    147 */	fmovd	%f298,%f296
/*    147 */	fmovd	%f42,%f294




/*    147 */	sxar2
/*    147 */	ldd,s	[%xg21],%f44
/*    147 */	fmovd	%f288,%f36



/*    147 */	sxar2
/*    147 */	sra	%xg19,1,%xg19
/*    147 */	add	%xg12,4,%xg24


/*    147 */	sxar2
/*    147 */	sra	%xg19,%g0,%xg19
/*    147 */	srl	%xg24,31,%xg25


/*    147 */	sxar2
/*    147 */	ldd,s	[%xg22],%f48
/*    147 */	ldd,s	[%xg23],%f50


/*    147 */	sxar2
/*    147 */	sllx	%xg19,3,%xg26
/*    147 */	add	%xg24,%xg25,%xg24


/*    147 */	sxar2
/*    147 */	ldd,s	[%xg18],%f54
/*    147 */	sub	%xg26,%xg19,%xg26


/*    147 */	sxar2
/*    147 */	sra	%xg24,1,%xg24
/*    147 */	std,s	%f34,[%xg9]


/*    147 */	fmovd	%f44,%f46


/*    147 */	sxar2
/*    147 */	sllx	%xg26,5,%xg26
/*    147 */	sra	%xg24,%g0,%xg24


/*    147 */	sxar2
/*    147 */	std,s	%f38,[%xg9+16]
/*    147 */	add	%xg26,%xg6,%xg27


/*    147 */	sxar1
/*    147 */	fmovd	%f48,%f302


/*    147 */	fmovd	%f50,%f52



/*    147 */	sxar2
/*    147 */	add	%xg26,%xg5,%xg28
/*    147 */	add	%xg26,%xg7,%xg29




/*    147 */	sxar2
/*    147 */	fmovd	%f300,%f48
/*    147 */	fmovd	%f54,%f308



/*    147 */	sxar2
/*    147 */	std,s	%f36,[%xg10]
/*    147 */	add	%xg26,%xg8,%xg26


/*    147 */	sxar2
/*    147 */	sllx	%xg24,3,%xg30
/*    147 */	fmovd	%f306,%f54



/*    147 */	sxar2
/*    147 */	std,s	%f40,[%xg10+16]
/*    147 */	add	%xg12,6,%o0


/*    147 */	sxar2
/*    147 */	sub	%xg30,%xg24,%xg30
/*    147 */	ldd,s	[%xg27],%f60


/*    147 */	sxar2
/*    147 */	ldd,s	[%xg28],%f64
/*    147 */	add	%xg3,256,%xg3


/*    147 */	sxar2
/*    147 */	sllx	%xg30,5,%xg30
/*    147 */	ldd,s	[%xg29],%f56


/*    147 */	sxar2
/*    147 */	ldd,s	[%xg26],%f66
/*    147 */	add	%xg30,%xg6,%xg31


/*    147 */	sxar2
/*    147 */	add	%xg30,%xg5,%g1
/*    147 */	std,s	%f46,[%xg9+64]


/*    147 */	sxar2
/*    147 */	add	%xg30,%xg7,%g2
/*    147 */	add	%xg30,%xg8,%xg30


/*    147 */	sxar2
/*    147 */	std,s	%f52,[%xg9+80]
/*    147 */	add	%xg1,256,%xg1

/*    147 */	sxar1
/*    147 */	sub	%xg0,4,%xg0



/*    147 */	fmovd	%f60,%f62



/*    147 */	sxar2
/*    147 */	fmovd	%f64,%f318
/*    147 */	ldd,s	[%xg31],%f68


/*    147 */	fmovd	%f56,%f58



/*    147 */	sxar2
/*    147 */	ldd,s	[%g1],%f72
/*    147 */	fmovd	%f66,%f314



/*    147 */	sxar2
/*    147 */	std,s	%f48,[%xg10+64]
/*    147 */	fmovd	%f316,%f64




/*    147 */	sxar2
/*    147 */	fmovd	%f312,%f66
/*    147 */	std,s	%f54,[%xg10+80]


/*    147 */	sxar2
/*    147 */	ldd,s	[%g2],%f74
/*    147 */	ldd,s	[%xg30],%f78




/*    147 */	sxar2
/*    147 */	fmovd	%f68,%f70
/*    147 */	fmovd	%f72,%f326




/*    147 */	sxar2
/*    147 */	fmovd	%f324,%f72
/*    147 */	std,s	%f62,[%xg9+128]



/*    147 */	sxar2
/*    147 */	std,s	%f58,[%xg9+144]
/*    147 */	fmovd	%f74,%f76




/*    147 */	sxar2
/*    147 */	fmovd	%f78,%f332
/*    147 */	std,s	%f64,[%xg10+128]



/*    147 */	sxar2
/*    147 */	fmovd	%f330,%f78
/*    147 */	std,s	%f66,[%xg10+144]


/*    147 */	sxar2
/*    147 */	std,s	%f70,[%xg9+192]
/*    147 */	std,s	%f76,[%xg9+208]


/*    147 */	sxar2
/*    147 */	add	%xg9,256,%xg9
/*    147 */	std,s	%f72,[%xg10+192]


/*    147 */	sxar2
/*    147 */	std,s	%f78,[%xg10+208]
/*    147 */	add	%xg10,256,%xg10

.L4988:


.L4987:


.L4990:


/*    156 */	sxar2
/*    156 */	srl	%o0,31,%xg15
/*    156 */	add	%o0,2,%xg16


/*    148 */	sxar2
/*    148 */	add	%xg15,%o0,%xg15
/*    148 */	srl	%xg16,31,%xg17


/*    148 */	sxar2
/*    148 */	sra	%xg15,1,%xg15
/*    148 */	add	%xg16,%xg17,%xg17


/*    148 */	sxar2
/*    148 */	sra	%xg15,%g0,%xg15
/*    148 */	sra	%xg17,1,%xg17


/*    148 */	sxar2
/*    148 */	sllx	%xg15,3,%xg18
/*    148 */	sra	%xg17,%g0,%xg17


/*    148 */	sxar2
/*    148 */	sub	%xg18,%xg15,%xg18
/*    148 */	sllx	%xg17,3,%xg19


/*    148 */	sxar2
/*    148 */	sllx	%xg18,5,%xg18
/*    148 */	sub	%xg19,%xg17,%xg19


/*    150 */	sxar2
/*    150 */	add	%xg18,%xg6,%xg20
/*    150 */	add	%xg18,%xg7,%xg22


/*    151 */	sxar2
/*    151 */	add	%xg18,%xg5,%xg21
/*    151 */	add	%xg18,%xg8,%xg18


/*    102 */	sxar2
/*    102 */	ldd,s	[%xg20],%f176
/*    102 */	ldd,s	[%xg22],%f182


/*    148 */	sxar2
/*    148 */	ldd,s	[%xg21],%f180
/*    148 */	sllx	%xg19,5,%xg19


/*    156 */	sxar2
/*    156 */	ldd,s	[%xg18],%f186
/*    156 */	add	%xg16,2,%xg16


/*    150 */	sxar2
/*    150 */	add	%xg19,%xg6,%xg23
/*    150 */	add	%xg19,%xg7,%xg25


/*    102 */	sxar2
/*    102 */	add	%xg19,%xg5,%xg24
/*    102 */	ldd,s	[%xg23],%f188


/*    148 */	sxar2
/*    148 */	ldd,s	[%xg25],%f194
/*    148 */	srl	%xg16,31,%xg26



/*    102 */	sxar2
/*    102 */	add	%xg19,%xg8,%xg19
/*    102 */	fmovd	%f432,%f178



/*    156 */	sxar2
/*    156 */	fmovd	%f438,%f184
/*    156 */	add	%xg16,2,%o0



/*    102 */	sxar2
/*    102 */	ldd,s	[%xg24],%f192
/*    102 */	fmovd	%f180,%f432





/*    148 */	sxar2
/*    148 */	fmovd	%f186,%f438
/*    148 */	add	%xg16,%xg26,%xg16


/*    102 */	sxar2
/*    102 */	fmovd	%f436,%f434
/*    102 */	ldd,s	[%xg19],%f198




/*    148 */	sxar2
/*    148 */	fmovd	%f442,%f440
/*    148 */	sra	%xg16,1,%xg16




/*    102 */	sxar2
/*    102 */	fmovd	%f444,%f190
/*    102 */	fmovd	%f450,%f196


/*    148 */	sxar2
/*    148 */	srl	%o0,31,%xg27
/*    148 */	sra	%xg16,%g0,%xg16


/*    148 */	sxar2
/*    148 */	add	%o0,%xg27,%xg27
/*    148 */	sllx	%xg16,3,%xg28



/*    102 */	sxar2
/*    102 */	fmovd	%f448,%f446
/*    102 */	fmovd	%f192,%f444




/*    148 */	sxar2
/*    148 */	sra	%xg27,1,%xg27
/*    148 */	sub	%xg28,%xg16,%xg28



/*    102 */	sxar2
/*    102 */	fmovd	%f454,%f452
/*    102 */	fmovd	%f198,%f450




/*    148 */	sxar2
/*    148 */	sra	%xg27,%g0,%xg27
/*    148 */	sllx	%xg28,5,%xg28


/*    148 */	sxar2
/*    148 */	std,s	%f176,[%xg9]
/*    148 */	add	%xg28,%xg6,%xg30


/*     25 */	sxar2
/*     25 */	add	%xg28,%xg7,%g1
/*     25 */	std,s	%f182,[%xg9+16]


/*    151 */	sxar2
/*    151 */	add	%xg28,%xg5,%xg31
/*    151 */	add	%xg28,%xg8,%xg28


/*    148 */	sxar2
/*    148 */	std,s	%f178,[%xg10]
/*    148 */	sllx	%xg27,3,%xg29



/*    148 */	sxar2
/*    148 */	std,s	%f184,[%xg10+16]
/*    148 */	sub	%xg29,%xg27,%xg29


/*    102 */	sxar2
/*    102 */	add	%xg3,256,%xg3
/*    102 */	ldd,s	[%xg30],%f200


/*    102 */	sxar2
/*    102 */	sllx	%xg29,5,%xg29
/*    102 */	ldd,s	[%g1],%f206


/*     24 */	sxar2
/*     24 */	add	%xg1,256,%xg1
/*     24 */	std,s	%f188,[%xg9+64]


/*    150 */	sxar2
/*    150 */	add	%xg29,%xg6,%g2
/*    150 */	add	%xg29,%xg7,%g4


/*    149 */	sxar2
/*    149 */	std,s	%f194,[%xg9+80]
/*    149 */	add	%xg29,%xg5,%g3


/*    102 */	sxar2
/*    102 */	add	%xg29,%xg8,%xg29
/*    102 */	ldd,s	[%xg31],%f204


/*    156 */	sxar2
/*    156 */	ldd,s	[%xg28],%f210
/*    156 */	subcc	%xg0,4,%xg0




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


/*    156 */	sxar2
/*    156 */	std,s	%f220,[%xg10+208]
/*    156 */	add	%xg10,256,%xg10

/*    156 */	bpos,pt	%icc, .L4990
/*    156 */	add	%o0,2,%o0


.L4986:


.L4861:

/*    147 */	sxar1
/*    147 */	addcc	%xg0,3,%xg0

/*    147 */	bneg	.L4856
	nop


.L4862:

/*    147 */	ldx	[%i0+2191],%g3

/*    147 */	ldx	[%i0+2199],%o4

/*    147 */	add	%g3,16,%g4

/*    147 */	add	%g3,32,%g5

/*    147 */	add	%g3,48,%o1

.L4869:

/*    148 */	srl	%o0,31,%o2

/*    154 */	sxar1
/*    154 */	add	%o4,%xg3,%o3

/*    148 */	add	%o2,%o0,%o2

/*    155 */	sxar1
/*    155 */	add	%o4,%xg1,%o5

/*    148 */	sra	%o2,1,%o2


/*    148 */	sra	%o2,%g0,%o2

/*    156 */	sxar1
/*    156 */	add	%xg1,64,%xg1

/*    148 */	sllx	%o2,3,%o7

/*    156 */	sxar1
/*    156 */	add	%xg3,64,%xg3

/*    148 */	sub	%o7,%o2,%o7

/*    156 */	sxar1
/*    156 */	subcc	%xg0,1,%xg0

/*    148 */	sllx	%o7,5,%o7


/*    149 */	sxar2
/*    149 */	add	%o7,%g3,%xg2
/*    149 */	add	%o7,%g4,%xg4

/*    150 */	sxar1
/*    150 */	add	%o7,%g5,%xg5

/*    151 */	add	%o7,%o1,%o7


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

/*    156 */	bpos,pt	%icc, .L4869
/*    156 */	add	%o0,2,%o0


.L4863:


.L4856:

/*    156 */
/*    156 */	ba	.L4853
	nop


.L4858:

/*    156 *//*    156 */	call	__mpc_obar
/*    156 */	ldx	[%fp+2199],%o0

/*    156 *//*    156 */	call	__mpc_obar
/*    156 */	ldx	[%fp+2199],%o0


.L4859:

/*    156 */	ret
	restore



.LLFE10:
	.size	_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_4,.-_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_4
	.type	_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_4,#function
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite4-k.cpp _ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_5 $"
	.section	".text"
	.align	64
_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_5:
.LLFB11:
.L4871:

/*    158 */	save	%sp,-2048,%sp
.LLCFI9:
/*    158 */	stx	%i0,[%fp+2175]
/*    158 */	stx	%i3,[%fp+2199]
/*    158 */	stx	%i0,[%fp+2175]

.L4872:

/*    158 *//*    158 */	ldsw	[%i0+2035],%g1
/*    158 */
/*    158 */
/*    158 */
/*    159 */	ldsw	[%i0+2179],%l0
/*    159 */	cmp	%l0,%g0
/*    159 */	ble	.L4886
/*    159 */	mov	%g0,%o0


.L4873:

/*    159 */	sxar1
/*    159 */	fzero,s	%f34

/*    159 */	sethi	%h44(.LR0.cnt.3),%g1

/*    159 */	sxar1
/*    159 */	sethi	%h44(.LR0.cnt.4),%xg0

/*    159 */	or	%g1,%m44(.LR0.cnt.3),%g1

/*    159 */	sxar1
/*    159 */	or	%xg0,%m44(.LR0.cnt.4),%xg0

/*    159 */	sllx	%g1,12,%g1

/*    159 */	sxar1
/*    159 */	sllx	%xg0,12,%xg0

/*    159 */	or	%g1,%l44(.LR0.cnt.3),%g1


/*    159 */	sxar2
/*    159 */	or	%xg0,%l44(.LR0.cnt.4),%xg0
/*    159 */	mov	1,%xg31

/*    159 */	sra	%l0,%g0,%l0


/*    159 */	sxar2
/*    159 */	ldd	[%g1],%f232
/*    159 */	ldd	[%g1],%f488



/*    159 */	sxar2
/*    159 */	ldd	[%xg0],%f234
/*    159 */	ldd	[%xg0],%f490



/*    159 */	sxar2
/*    ??? */	std,s	%f34,[%fp+223]
/*    159 */	stx	%xg31,[%fp+2031]


/*    159 */	sxar2
/*    ??? */	std,s	%f232,[%fp+255]
/*    ??? */	std,s	%f234,[%fp+239]

.L4903:

/*    159 */	add	%fp,2039,%l1

/*    159 */	mov	1,%l5

/*    159 */	add	%fp,2023,%l2

/*    159 */	add	%fp,2031,%l3

/*    159 */	sra	%l5,%g0,%l4

.L4875:

/*    159 */	sra	%o0,%g0,%o0

/*    159 */	stx	%g0,[%sp+2223]

/*    159 */	mov	4,%o2

/*    159 */	mov	%g0,%o3

/*    159 */	mov	%l0,%o1

/*    159 */	mov	%l1,%o4


/*    159 */	stx	%g0,[%sp+2231]

/*    159 */	stx	%l3,[%sp+2239]


/*    159 */	sxar2
/*    159 */	ldx	[%fp+2199],%xg29
/*    159 */	stx	%xg29,[%sp+2247]

/*    159 */	call	__mpc_ostd_th
/*    159 */	mov	%l2,%o5
/*    159 */	sxar2
/*    159 */	ldx	[%fp+2031],%xg30
/*    159 */	cmp	%xg30,%g0
/*    159 */	ble,pn	%xcc, .L4886
	nop


.L4876:

/*    159 */	ldx	[%fp+2039],%o0


/*    159 */	sxar2
/*    159 */	ldx	[%fp+2023],%xg0
/*    159 */	ldd	[%i0+2183],%f74



/*    159 */	sxar2
/*    159 */	ldd	[%i0+2183],%f330
/*    159 */	ldsw	[%i0+2179],%xg9


/*    159 */	sxar2
/*    159 */	ldx	[%i0+2199],%xg5
/*    159 */	ldx	[%i0+2207],%xg19

/*    159 */	sra	%o0,%g0,%o0


/*    159 */	sxar2
/*    159 */	sra	%xg0,%g0,%xg0
/*    159 */	sub	%xg0,%o0,%xg0


/*    159 */	sxar2
/*    159 */	sra	%o0,%g0,%xg1
/*    159 */	sra	%xg0,1,%xg2


/*    159 */	sxar2
/*    159 */	sllx	%xg1,5,%xg3
/*    159 */	srl	%xg2,30,%xg2


/*    159 */	sxar2
/*    159 */	sllx	%xg1,3,%xg1
/*    159 */	add	%xg0,%xg2,%xg0


/*    159 */	sxar2
/*    159 */	add	%xg5,32,%xg4
/*    159 */	sra	%xg0,2,%xg0


/*    159 */	sxar2
/*    159 */	add	%xg0,1,%xg0
/*    159 */	sra	%xg0,%g0,%xg0


/*    159 */	sxar2
/*    159 */	sub	%l4,%xg0,%xg0
/*    159 */	srax	%xg0,32,%xg6


/*    159 */	sxar2
/*    159 */	and	%xg0,%xg6,%xg0
/*    159 */	sub	%l5,%xg0,%xg0

.L4877:


/*    166 */	sxar2
/*    166 */	add	%xg5,%xg3,%xg7
/*    166 */	cmp	%xg9,%g0


/*    161 */	sxar2
/*    ??? */	ldd,s	[%fp+223],%f32
/*    161 */	ldd	[%xg7],%f250



/*    161 */	sxar2
/*    161 */	ldd	[%xg7+32],%f506
/*    161 */	ldd	[%xg7+64],%f38



/*     37 */	sxar2
/*     37 */	ldd	[%xg7+96],%f294
/*     37 */	std,s	%f250,[%fp+1135]


/*    162 */	sxar2
/*    162 */	std,s	%f38,[%fp+1151]
/*    162 */	ldd	[%xg7+8],%f252



/*    162 */	sxar2
/*    162 */	ldd	[%xg7+40],%f508
/*    162 */	ldd	[%xg7+72],%f46



/*     37 */	sxar2
/*     37 */	ldd	[%xg7+104],%f302
/*     37 */	std,s	%f252,[%fp+1167]


/*    163 */	sxar2
/*    163 */	std,s	%f46,[%fp+1183]
/*    163 */	ldd	[%xg7+16],%f254



/*    163 */	sxar2
/*    163 */	ldd	[%xg7+48],%f510
/*    163 */	ldd	[%xg7+80],%f56



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

/*    166 */	ble	.L4883
	nop


.L4879:


/*    178 */	sxar2
/*    178 */	ldd,s	[%fp+1135],%f32
/*    178 */	mov	%g0,%xg10


/*    178 */	sxar2
/*    ??? */	ldd,s	[%fp+223],%f186
/*    178 */	subcc	%xg9,2,%xg8


/*    166 */	sxar2
/*    166 */	ldd,s	[%fp+1263],%f184
/*    166 */	ldd,s	[%fp+1167],%f42


/*    166 */	sxar2
/*    166 */	ldd,s	[%fp+1199],%f50
/*    166 */	ldd,s	[%fp+1231],%f72

/*    178 */	bneg	.L4889
	nop


.L4892:


/*    166 */	sxar2
/*    166 */	ldx	[%i0+2199],%xg12
/*    166 */	cmp	%xg8,14

/*    166 */	bl	.L4997
	nop


.L4993:


.L5000:


/*    166 */	sxar2
/*    166 */	add	%xg12,%xg10,%xg11
/*    166 */	add	%xg4,%xg10,%xg13


/*    166 */	sxar2
/*    ??? */	ldd,s	[%fp+255],%f158
/*    166 */	ldd,s	[%xg11],%f36


/*    166 */	sxar2
/*    166 */	ldd,s	[%xg11+16],%f52
/*    166 */	add	%xg10,64,%xg14


/*    166 */	sxar2
/*    166 */	add	%xg10,128,%xg10
/*    166 */	add	%xg12,%xg14,%xg15


/*    166 */	sxar2
/*    166 */	ldd,s	[%xg13],%f60
/*    166 */	ldd,s	[%xg13+16],%f68


/*    166 */	sxar2
/*    166 */	add	%xg4,%xg14,%xg14
/*    166 */	ldd,s	[%xg15],%f78


/*    166 */	sxar2
/*    166 */	add	%xg12,%xg10,%xg16
/*    ??? */	ldd,s	[%fp+239],%f160


/*    166 */	sxar2
/*    166 */	ldd,s	[%xg15+16],%f86
/*    166 */	ldd,s	[%xg14],%f92


/*    166 */	sxar2
/*    166 */	ldd,s	[%xg16],%f100
/*    166 */	fnmsubd,sc	%f36,%f158,%f32,%f34


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f36,%f158,%f38,%f40
/*    166 */	ldd,s	[%xg14+16],%f106


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f292,%f158,%f42,%f44
/*    166 */	fnmsubd,sc	%f292,%f158,%f46,%f36


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f52,%f158,%f50,%f48
/*    166 */	fnmsubd,sc	%f52,%f158,%f56,%f54


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f60,%f158,%f32,%f58
/*    166 */	fnmsubd,sc	%f60,%f158,%f38,%f62


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f316,%f158,%f42,%f64
/*    166 */	fnmsubd,sc	%f316,%f158,%f46,%f60


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f68,%f158,%f50,%f66
/*    166 */	fnmsubd,sc	%f68,%f158,%f56,%f70


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f34,%f34,%f72,%f34
/*    166 */	fmaddd,s	%f40,%f40,%f74,%f40


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f78,%f158,%f32,%f76
/*    166 */	fnmsubd,sc	%f78,%f158,%f38,%f80


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f334,%f158,%f42,%f82
/*    166 */	fnmsubd,sc	%f334,%f158,%f46,%f78


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f58,%f58,%f72,%f58
/*    166 */	fmaddd,s	%f62,%f62,%f74,%f62


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f86,%f158,%f50,%f84
/*    166 */	fnmsubd,sc	%f86,%f158,%f56,%f88


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f92,%f158,%f32,%f90
/*    166 */	fnmsubd,sc	%f92,%f158,%f38,%f94


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f44,%f44,%f34,%f44
/*    166 */	fmaddd,s	%f36,%f36,%f40,%f36


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f76,%f76,%f72,%f76
/*    166 */	fmaddd,s	%f80,%f80,%f74,%f80


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f348,%f158,%f42,%f96
/*    166 */	fnmsubd,sc	%f348,%f158,%f46,%f92


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f64,%f64,%f58,%f64
/*    166 */	fmaddd,s	%f60,%f60,%f62,%f60


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f100,%f158,%f32,%f98
/*    166 */	fnmsubd,sc	%f100,%f158,%f38,%f102


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f90,%f90,%f72,%f90
/*    166 */	fmaddd,s	%f94,%f94,%f74,%f94


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f48,%f48,%f44,%f48
/*    166 */	fmaddd,s	%f54,%f54,%f36,%f54


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f82,%f82,%f76,%f82
/*    166 */	fmaddd,s	%f78,%f78,%f80,%f78


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f106,%f158,%f50,%f104
/*    166 */	fnmsubd,sc	%f106,%f158,%f56,%f108


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f66,%f66,%f64,%f66
/*    166 */	fmaddd,s	%f70,%f70,%f60,%f70


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f96,%f96,%f90,%f96
/*    166 */	fmaddd,s	%f92,%f92,%f94,%f92


/*    166 */	sxar2
/*    166 */	frsqrtad,s	%f48,%f110
/*    166 */	frsqrtad,s	%f54,%f112


/*    166 */	sxar2
/*    166 */	fmuld,s	%f48,%f160,%f114
/*    166 */	fmuld,s	%f54,%f160,%f116


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f84,%f84,%f82,%f84
/*    166 */	fmaddd,s	%f88,%f88,%f78,%f88


/*    166 */	sxar2
/*    166 */	frsqrtad,s	%f66,%f118
/*    166 */	frsqrtad,s	%f70,%f120


/*    166 */	sxar2
/*    166 */	fmuld,s	%f66,%f160,%f122
/*    166 */	fmuld,s	%f70,%f160,%f124


/*    166 */	sxar2
/*    166 */	fmuld,s	%f110,%f110,%f126
/*    166 */	fmuld,s	%f112,%f112,%f128


/*    166 */	sxar2
/*    166 */	frsqrtad,s	%f84,%f130
/*    166 */	frsqrtad,s	%f88,%f132


/*    166 */	sxar2
/*    166 */	fmuld,s	%f118,%f118,%f134
/*    166 */	fmuld,s	%f120,%f120,%f136


/*    166 */	sxar2
/*    166 */	fmuld,s	%f84,%f160,%f138
/*    166 */	fmuld,s	%f88,%f160,%f140


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f114,%f126,%f160,%f126
/*    166 */	fnmsubd,s	%f116,%f128,%f160,%f128


/*    166 */	sxar2
/*    166 */	fmuld,s	%f130,%f130,%f142
/*    166 */	fmuld,s	%f132,%f132,%f144


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f122,%f134,%f160,%f134
/*    166 */	fnmsubd,s	%f124,%f136,%f160,%f136


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f110,%f126,%f110,%f110
/*    166 */	fmaddd,s	%f112,%f128,%f112,%f112


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f118,%f134,%f118,%f118
/*    166 */	fmaddd,s	%f120,%f136,%f120,%f120


/*    166 */	sxar2
/*    166 */	fmuld,s	%f110,%f110,%f146
/*    166 */	fmuld,s	%f112,%f112,%f148


/*    166 */	sxar2
/*    166 */	fmuld,s	%f118,%f118,%f150
/*    166 */	fmuld,s	%f120,%f120,%f152


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f114,%f146,%f160,%f146
/*    166 */	fnmsubd,s	%f116,%f148,%f160,%f148


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f122,%f150,%f160,%f150
/*    166 */	fnmsubd,s	%f124,%f152,%f160,%f152


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f110,%f146,%f110,%f110
/*    166 */	fmaddd,s	%f112,%f148,%f112,%f112


/*    166 */	sxar2
/*    166 */	fmuld,s	%f110,%f110,%f154
/*    166 */	fmuld,s	%f112,%f112,%f156

.L4881:


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f356,%f158,%f42,%f36
/*    166 */	fnmsubd,sc	%f356,%f158,%f46,%f100


/*    166 */	sxar2
/*    166 */	ldd,s	[%xg16+16],%f164
/*    166 */	fmaddd,s	%f98,%f98,%f72,%f98


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f104,%f104,%f96,%f104
/*    166 */	fmaddd,s	%f102,%f102,%f74,%f102


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f108,%f108,%f92,%f108
/*    166 */	fmaddd,s	%f118,%f150,%f118,%f118


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f120,%f152,%f120,%f120
/*    166 */	fnmsubd,s	%f138,%f142,%f160,%f142


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f114,%f154,%f160,%f114
/*    166 */	fnmsubd,s	%f140,%f144,%f160,%f144


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f116,%f156,%f160,%f116
/*    166 */	fnmsubd,sc	%f164,%f158,%f50,%f162


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f164,%f158,%f56,%f166
/*    166 */	fmaddd,s	%f36,%f36,%f98,%f36


/*    166 */	sxar2
/*    166 */	frsqrtad,s	%f104,%f168
/*    166 */	fmaddd,s	%f100,%f100,%f102,%f100


/*    166 */	sxar2
/*    166 */	frsqrtad,s	%f108,%f170
/*    166 */	add	%xg10,64,%xg20


/*    166 */	sxar2
/*    166 */	fmuld,s	%f118,%f118,%f172
/*    166 */	fmuld,s	%f120,%f120,%f174


/*    166 */	sxar2
/*    166 */	add	%xg12,%xg20,%xg21
/*    166 */	fmaddd,s	%f130,%f142,%f130,%f130


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f110,%f114,%f110,%f110
/*    166 */	ldd,s	[%xg21],%f190


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f132,%f144,%f132,%f132
/*    166 */	fmaddd,s	%f112,%f116,%f112,%f112


/*    166 */	sxar2
/*    166 */	fcmplted,s	%f72,%f48,%f48
/*    166 */	fcmplted,s	%f74,%f54,%f54


/*    166 */	sxar2
/*    166 */	fmuld,s	%f104,%f160,%f176
/*    166 */	fmuld,s	%f168,%f168,%f178


/*    166 */	sxar2
/*    166 */	fmuld,s	%f108,%f160,%f180
/*    166 */	fmuld,s	%f170,%f170,%f182


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f122,%f172,%f160,%f122
/*    166 */	fnmsubd,s	%f124,%f174,%f160,%f124


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f190,%f158,%f32,%f188
/*    166 */	fnmsubd,sc	%f190,%f158,%f38,%f192


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f162,%f162,%f36,%f162
/*    166 */	fmuld,s	%f130,%f130,%f194


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f166,%f166,%f100,%f166
/*    166 */	fmuld,s	%f132,%f132,%f196


/*    166 */	sxar2
/*    166 */	add	%xg4,%xg10,%xg22
/*    166 */	fand,s	%f110,%f48,%f110


/*    166 */	sxar2
/*    166 */	fand,s	%f112,%f54,%f112
/*    166 */	ldd,s	[%xg22],%f204


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f176,%f178,%f160,%f178
/*    166 */	fnmsubd,s	%f180,%f182,%f160,%f182


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f118,%f122,%f118,%f118
/*    166 */	fmaddd,s	%f120,%f124,%f120,%f120


/*    166 */	sxar2
/*    166 */	fcmplted,s	%f72,%f66,%f66
/*    166 */	fcmplted,s	%f74,%f70,%f70


/*    166 */	sxar2
/*    166 */	frsqrtad,s	%f162,%f198
/*    166 */	fnmsubd,s	%f138,%f194,%f160,%f194


/*    166 */	sxar2
/*    166 */	fmuld,s	%f162,%f160,%f200
/*    166 */	fnmsubd,sc	%f204,%f158,%f32,%f202


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f140,%f196,%f160,%f196
/*    166 */	fnmsubd,sc	%f204,%f158,%f38,%f206


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f168,%f178,%f168,%f168
/*    166 */	fmaddd,s	%f170,%f182,%f170,%f170


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f308,%f110,%f184,%f110
/*    166 */	fnmsubd,sc	%f308,%f112,%f186,%f52


/*    166 */	sxar2
/*    166 */	fand,s	%f118,%f66,%f118
/*    166 */	fand,s	%f120,%f70,%f120


/*    166 */	sxar2
/*    166 */	fmuld,s	%f198,%f198,%f208
/*    166 */	fmaddd,s	%f130,%f194,%f130,%f130


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f460,%f158,%f42,%f210
/*    166 */	fmaddd,s	%f202,%f202,%f72,%f202


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f460,%f158,%f46,%f204
/*    166 */	ldd,s	[%xg22+16],%f186


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f206,%f206,%f74,%f206
/*    166 */	fmuld,s	%f168,%f168,%f212


/*    166 */	sxar2
/*    166 */	fmuld,s	%f170,%f170,%f214
/*    166 */	frsqrtad,s	%f166,%f216


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f132,%f196,%f132,%f132
/*    166 */	fnmsubd,sc	%f324,%f118,%f110,%f118


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f324,%f120,%f52,%f68
/*    166 */	fmuld,s	%f130,%f130,%f218


/*    166 */	sxar2
/*    166 */	fmuld,s	%f166,%f160,%f220
/*    166 */	fnmsubd,sc	%f186,%f158,%f50,%f222


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f210,%f210,%f202,%f210
/*    166 */	fnmsubd,sc	%f186,%f158,%f56,%f224


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f204,%f204,%f206,%f204
/*    166 */	fnmsubd,s	%f176,%f212,%f160,%f212


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f180,%f214,%f160,%f214
/*    166 */	fmuld,s	%f216,%f216,%f226


/*    166 */	sxar2
/*    166 */	fmuld,s	%f132,%f132,%f228
/*    166 */	fnmsubd,sc	%f446,%f158,%f42,%f230


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f446,%f158,%f46,%f190
/*    166 */	ldd,s	[%xg21+16],%f52


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f188,%f188,%f72,%f188
/*    166 */	fmaddd,s	%f222,%f222,%f210,%f222


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f192,%f192,%f74,%f192
/*    166 */	fmaddd,s	%f224,%f224,%f204,%f224


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f168,%f212,%f168,%f168
/*    166 */	fmaddd,s	%f170,%f214,%f170,%f170


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f200,%f208,%f160,%f208
/*    166 */	fnmsubd,s	%f138,%f218,%f160,%f138


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f220,%f226,%f160,%f226
/*    166 */	fnmsubd,s	%f140,%f228,%f160,%f140


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f52,%f158,%f50,%f48
/*    166 */	fnmsubd,sc	%f52,%f158,%f56,%f54


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f230,%f230,%f188,%f230
/*    166 */	frsqrtad,s	%f222,%f184


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f190,%f190,%f192,%f190
/*    166 */	frsqrtad,s	%f224,%f232


/*    166 */	sxar2
/*    166 */	add	%xg10,128,%xg23
/*    166 */	fmuld,s	%f168,%f168,%f234


/*    166 */	sxar2
/*    166 */	fmuld,s	%f170,%f170,%f236
/*    166 */	add	%xg12,%xg23,%xg24


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f198,%f208,%f198,%f198
/*    166 */	fmaddd,s	%f130,%f138,%f130,%f130


/*    166 */	sxar2
/*    166 */	ldd,s	[%xg24],%f248
/*    166 */	fmaddd,s	%f216,%f226,%f216,%f216


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f132,%f140,%f132,%f132
/*    166 */	fcmplted,s	%f72,%f84,%f84


/*    166 */	sxar2
/*    166 */	fcmplted,s	%f74,%f88,%f88
/*    166 */	fmuld,s	%f222,%f160,%f238


/*    166 */	sxar2
/*    166 */	fmuld,s	%f184,%f184,%f240
/*    166 */	fmuld,s	%f224,%f160,%f242


/*    166 */	sxar2
/*    166 */	fmuld,s	%f232,%f232,%f244
/*    166 */	fnmsubd,s	%f176,%f234,%f160,%f176


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f180,%f236,%f160,%f180
/*    166 */	fnmsubd,sc	%f248,%f158,%f32,%f246


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f248,%f158,%f38,%f250
/*    166 */	fmaddd,s	%f48,%f48,%f230,%f48


/*    166 */	sxar2
/*    166 */	fmuld,s	%f198,%f198,%f252
/*    166 */	fmaddd,s	%f54,%f54,%f190,%f54


/*    166 */	sxar2
/*    166 */	fmuld,s	%f216,%f216,%f254
/*    166 */	add	%xg4,%xg20,%xg20


/*    166 */	sxar2
/*    166 */	fand,s	%f130,%f84,%f130
/*    166 */	fand,s	%f132,%f88,%f132


/*    166 */	sxar2
/*    166 */	ldd,s	[%xg20],%f36
/*    166 */	fnmsubd,s	%f238,%f240,%f160,%f240


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f242,%f244,%f160,%f244
/*    166 */	fmaddd,s	%f168,%f176,%f168,%f168


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f170,%f180,%f170,%f170
/*    166 */	fcmplted,s	%f72,%f104,%f104


/*    166 */	sxar2
/*    166 */	fcmplted,s	%f74,%f108,%f108
/*    166 */	frsqrtad,s	%f48,%f110


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f200,%f252,%f160,%f252
/*    166 */	fmuld,s	%f48,%f160,%f114


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f36,%f158,%f32,%f34
/*    166 */	fnmsubd,s	%f220,%f254,%f160,%f254


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f36,%f158,%f38,%f40
/*    166 */	fmaddd,s	%f184,%f240,%f184,%f184


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f232,%f244,%f232,%f232
/*    166 */	fnmsubd,sc	%f342,%f130,%f118,%f130


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f342,%f132,%f68,%f86
/*    166 */	fand,s	%f168,%f104,%f168


/*    166 */	sxar2
/*    166 */	fand,s	%f170,%f108,%f170
/*    166 */	fmuld,s	%f110,%f110,%f44


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f198,%f252,%f198,%f198
/*    166 */	fnmsubd,sc	%f292,%f158,%f42,%f58


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f34,%f34,%f72,%f34
/*    166 */	fnmsubd,sc	%f292,%f158,%f46,%f36


/*    166 */	sxar2
/*    166 */	ldd,s	[%xg20+16],%f68
/*    166 */	fmaddd,s	%f40,%f40,%f74,%f40


/*    166 */	sxar2
/*    166 */	fmuld,s	%f184,%f184,%f60
/*    166 */	fmuld,s	%f232,%f232,%f62


/*    166 */	sxar2
/*    166 */	frsqrtad,s	%f54,%f112
/*    166 */	fmaddd,s	%f216,%f254,%f216,%f216


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f362,%f168,%f130,%f168
/*    166 */	fnmsubd,sc	%f362,%f170,%f86,%f106


/*    166 */	sxar2
/*    166 */	fmuld,s	%f198,%f198,%f64
/*    166 */	fmuld,s	%f54,%f160,%f116


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f68,%f158,%f50,%f66
/*    166 */	fmaddd,s	%f58,%f58,%f34,%f58


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f68,%f158,%f56,%f70
/*    166 */	fmaddd,s	%f36,%f36,%f40,%f36


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f238,%f60,%f160,%f60
/*    166 */	fnmsubd,s	%f242,%f62,%f160,%f62


/*    166 */	sxar2
/*    166 */	fmuld,s	%f112,%f112,%f76
/*    166 */	fmuld,s	%f216,%f216,%f78


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f504,%f158,%f42,%f80
/*    166 */	fnmsubd,sc	%f504,%f158,%f46,%f248


/*    166 */	sxar2
/*    166 */	ldd,s	[%xg24+16],%f86
/*    166 */	fmaddd,s	%f246,%f246,%f72,%f246


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f66,%f66,%f58,%f66
/*    166 */	fmaddd,s	%f250,%f250,%f74,%f250


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f70,%f70,%f36,%f70
/*    166 */	fmaddd,s	%f184,%f60,%f184,%f184


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f232,%f62,%f232,%f232
/*    166 */	fnmsubd,s	%f114,%f44,%f160,%f44


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f200,%f64,%f160,%f200
/*    166 */	fnmsubd,s	%f116,%f76,%f160,%f76


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f220,%f78,%f160,%f220
/*    166 */	fnmsubd,sc	%f86,%f158,%f50,%f84


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f86,%f158,%f56,%f88
/*    166 */	fmaddd,s	%f80,%f80,%f246,%f80


/*    166 */	sxar2
/*    166 */	frsqrtad,s	%f66,%f118
/*    166 */	fmaddd,s	%f248,%f248,%f250,%f248


/*    166 */	sxar2
/*    166 */	frsqrtad,s	%f70,%f120
/*    166 */	add	%xg10,192,%xg10


/*    166 */	sxar2
/*    166 */	fmuld,s	%f184,%f184,%f82
/*    166 */	fmuld,s	%f232,%f232,%f90


/*    166 */	sxar2
/*    166 */	add	%xg12,%xg10,%xg16
/*    166 */	fmaddd,s	%f110,%f44,%f110,%f110


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f198,%f200,%f198,%f198
/*    166 */	ldd,s	[%xg16],%f100


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f112,%f76,%f112,%f112
/*    166 */	fmaddd,s	%f216,%f220,%f216,%f216


/*    166 */	sxar2
/*    166 */	fcmplted,s	%f72,%f162,%f162
/*    166 */	fcmplted,s	%f74,%f166,%f166


/*    166 */	sxar2
/*    166 */	fmuld,s	%f66,%f160,%f122
/*    166 */	fmuld,s	%f118,%f118,%f94


/*    166 */	sxar2
/*    166 */	fmuld,s	%f70,%f160,%f124
/*    166 */	fmuld,s	%f120,%f120,%f96


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f238,%f82,%f160,%f238
/*    166 */	fnmsubd,s	%f242,%f90,%f160,%f242


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f100,%f158,%f32,%f98
/*    166 */	fnmsubd,sc	%f100,%f158,%f38,%f102


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f84,%f84,%f80,%f84
/*    166 */	fmuld,s	%f110,%f110,%f104


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f88,%f88,%f248,%f88
/*    166 */	fmuld,s	%f112,%f112,%f108


/*    166 */	sxar2
/*    166 */	add	%xg4,%xg23,%xg23
/*    166 */	fand,s	%f198,%f162,%f198


/*    166 */	sxar2
/*    166 */	fand,s	%f216,%f166,%f216
/*    166 */	ldd,s	[%xg23],%f92


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f122,%f94,%f160,%f94
/*    166 */	fnmsubd,s	%f124,%f96,%f160,%f96


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f184,%f238,%f184,%f184
/*    166 */	fmaddd,s	%f232,%f242,%f232,%f232


/*    166 */	sxar2
/*    166 */	fcmplted,s	%f72,%f222,%f222
/*    166 */	fcmplted,s	%f74,%f224,%f224


/*    166 */	sxar2
/*    166 */	frsqrtad,s	%f84,%f130
/*    166 */	fnmsubd,s	%f114,%f104,%f160,%f104


/*    166 */	sxar2
/*    166 */	fmuld,s	%f84,%f160,%f138
/*    166 */	fnmsubd,sc	%f92,%f158,%f32,%f126


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f116,%f108,%f160,%f108
/*    166 */	fnmsubd,sc	%f92,%f158,%f38,%f128


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f118,%f94,%f118,%f118
/*    166 */	fmaddd,s	%f120,%f96,%f120,%f120


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f420,%f198,%f168,%f198
/*    166 */	fnmsubd,sc	%f420,%f216,%f106,%f164


/*    166 */	sxar2
/*    166 */	fand,s	%f184,%f222,%f184
/*    166 */	fand,s	%f232,%f224,%f232


/*    166 */	sxar2
/*    166 */	fmuld,s	%f130,%f130,%f142
/*    166 */	fmaddd,s	%f110,%f104,%f110,%f110


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f348,%f158,%f42,%f96
/*    166 */	fmaddd,s	%f126,%f126,%f72,%f126


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f348,%f158,%f46,%f92
/*    166 */	ldd,s	[%xg23+16],%f106


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f128,%f128,%f74,%f128
/*    166 */	fmuld,s	%f118,%f118,%f150


/*    166 */	sxar2
/*    166 */	fmuld,s	%f120,%f120,%f152
/*    166 */	frsqrtad,s	%f88,%f132


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f112,%f108,%f112,%f112
/*    166 */	fnmsubd,sc	%f442,%f184,%f198,%f184


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f442,%f232,%f164,%f186
/*    166 */	fmuld,s	%f110,%f110,%f154


/*    166 */	sxar2
/*    166 */	fmuld,s	%f88,%f160,%f140
/*    166 */	fnmsubd,sc	%f106,%f158,%f50,%f104


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f96,%f96,%f126,%f96
/*    166 */	fnmsubd,sc	%f106,%f158,%f56,%f108


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f92,%f92,%f128,%f92
/*    166 */	fnmsubd,s	%f122,%f150,%f160,%f150


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f124,%f152,%f160,%f152
/*    166 */	fmuld,s	%f132,%f132,%f144


/*    166 */	sxar2
/*    166 */	fmuld,s	%f112,%f112,%f156
/*    166 */	sub	%xg8,6,%xg8

/*    166 */	sxar1
/*    166 */	cmp	%xg8,15

/*    166 */	bge,pt	%icc, .L4881
	nop


.L5001:


/*    166 */	sxar2
/*    ??? */	ldd,s	[%fp+255],%f248
/*    166 */	ldd,s	[%xg16+16],%f162


/*    166 */	sxar2
/*    166 */	add	%xg4,%xg10,%xg17
/*    166 */	fmaddd,s	%f98,%f98,%f72,%f98


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f104,%f104,%f96,%f104
/*    166 */	ldd,s	[%xg17],%f168


/*    166 */	sxar2
/*    166 */	ldd,s	[%xg17+16],%f188
/*    166 */	fmaddd,s	%f102,%f102,%f74,%f102


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f108,%f108,%f92,%f108
/*    166 */	add	%xg10,64,%xg10


/*    166 */	sxar2
/*    ??? */	ldd,s	[%fp+239],%f236
/*    166 */	fcmplted,s	%f72,%f48,%f48


/*    166 */	sxar2
/*    166 */	sub	%xg8,6,%xg8
/*    166 */	fmaddd,s	%f118,%f150,%f118,%f118


/*    166 */	sxar2
/*    166 */	fcmplted,s	%f74,%f54,%f54
/*    166 */	fnmsubd,sc	%f356,%f248,%f42,%f158


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f356,%f248,%f46,%f100
/*    166 */	fnmsubd,sc	%f162,%f248,%f50,%f160


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f168,%f248,%f32,%f166
/*    166 */	fnmsubd,sc	%f168,%f248,%f38,%f170


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f424,%f248,%f42,%f172
/*    166 */	fnmsubd,sc	%f162,%f248,%f56,%f164


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f424,%f248,%f46,%f168
/*    166 */	fnmsubd,s	%f138,%f142,%f236,%f142


/*    166 */	sxar2
/*    166 */	frsqrtad,s	%f104,%f174
/*    166 */	frsqrtad,s	%f108,%f176


/*    166 */	sxar2
/*    166 */	fmuld,s	%f104,%f236,%f178
/*    166 */	fmaddd,s	%f158,%f158,%f98,%f158


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f100,%f100,%f102,%f100
/*    166 */	fnmsubd,sc	%f188,%f248,%f50,%f182


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f166,%f166,%f72,%f166
/*    166 */	fnmsubd,sc	%f188,%f248,%f56,%f190


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f170,%f170,%f74,%f170
/*    166 */	fnmsubd,s	%f140,%f144,%f236,%f144


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f114,%f154,%f236,%f114
/*    166 */	fmuld,s	%f108,%f236,%f180


/*    166 */	sxar2
/*    166 */	fmuld,s	%f174,%f174,%f192
/*    166 */	fmaddd,s	%f130,%f142,%f130,%f130


/*    166 */	sxar2
/*    166 */	fmuld,s	%f176,%f176,%f194
/*    166 */	fmaddd,s	%f160,%f160,%f158,%f160


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f164,%f164,%f100,%f164
/*    166 */	fnmsubd,s	%f116,%f156,%f236,%f116


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f172,%f172,%f166,%f172
/*    166 */	fmaddd,s	%f120,%f152,%f120,%f120


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f168,%f168,%f170,%f168
/*    166 */	fmaddd,s	%f132,%f144,%f132,%f132


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f110,%f114,%f110,%f110
/*    166 */	fnmsubd,s	%f178,%f192,%f236,%f192


/*    166 */	sxar2
/*    166 */	fmuld,s	%f118,%f118,%f208
/*    166 */	fmuld,s	%f130,%f130,%f196


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f180,%f194,%f236,%f194
/*    166 */	frsqrtad,s	%f160,%f200


/*    166 */	sxar2
/*    166 */	frsqrtad,s	%f164,%f202
/*    166 */	fmuld,s	%f160,%f236,%f204


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f182,%f182,%f172,%f182
/*    166 */	fmuld,s	%f164,%f236,%f206


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f190,%f190,%f168,%f190
/*    166 */	fmaddd,s	%f112,%f116,%f112,%f112


/*    166 */	sxar2
/*    166 */	fand,s	%f110,%f48,%f110
/*    166 */	fmuld,s	%f132,%f132,%f198


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f174,%f192,%f174,%f174
/*    166 */	fnmsubd,s	%f138,%f196,%f236,%f196


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f176,%f194,%f176,%f176
/*    166 */	fmuld,s	%f200,%f200,%f212


/*    166 */	sxar2
/*    166 */	fmuld,s	%f202,%f202,%f214
/*    166 */	frsqrtad,s	%f182,%f216


/*    166 */	sxar2
/*    166 */	fmuld,s	%f182,%f236,%f220
/*    166 */	frsqrtad,s	%f190,%f218


/*    166 */	sxar2
/*    166 */	fmuld,s	%f190,%f236,%f222
/*    166 */	fand,s	%f112,%f54,%f112


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f308,%f110,%f184,%f110
/*    166 */	fnmsubd,s	%f140,%f198,%f236,%f198


/*    166 */	sxar2
/*    166 */	fmuld,s	%f174,%f174,%f224
/*    166 */	fmuld,s	%f176,%f176,%f226


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f130,%f196,%f130,%f130
/*    166 */	fnmsubd,s	%f204,%f212,%f236,%f212


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f206,%f214,%f236,%f214
/*    166 */	fmuld,s	%f120,%f120,%f210


/*    166 */	sxar2
/*    166 */	fmuld,s	%f216,%f216,%f228
/*    166 */	fnmsubd,s	%f122,%f208,%f236,%f122


/*    166 */	sxar2
/*    166 */	fmuld,s	%f218,%f218,%f230
/*    166 */	fcmplted,s	%f72,%f66,%f66


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f308,%f112,%f186,%f52
/*    166 */	fmaddd,s	%f132,%f198,%f132,%f132


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f178,%f224,%f236,%f224
/*    166 */	fnmsubd,s	%f180,%f226,%f236,%f226


/*    166 */	sxar2
/*    166 */	fmuld,s	%f130,%f130,%f232
/*    166 */	fmaddd,s	%f200,%f212,%f200,%f200


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f202,%f214,%f202,%f202
/*    166 */	fnmsubd,s	%f124,%f210,%f236,%f124


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f220,%f228,%f236,%f228
/*    166 */	fmaddd,s	%f118,%f122,%f118,%f118


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f222,%f230,%f236,%f230
/*    166 */	fcmplted,s	%f74,%f70,%f70


/*    166 */	sxar2
/*    166 */	fcmplted,s	%f72,%f84,%f84
/*    166 */	fmuld,s	%f132,%f132,%f234


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f174,%f224,%f174,%f174
/*    166 */	fmaddd,s	%f176,%f226,%f176,%f176


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f138,%f232,%f236,%f138
/*    166 */	fmuld,s	%f200,%f200,%f238


/*    166 */	sxar2
/*    166 */	fmuld,s	%f202,%f202,%f240
/*    166 */	fmaddd,s	%f120,%f124,%f120,%f120


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f216,%f228,%f216,%f184
/*    166 */	fand,s	%f118,%f66,%f118


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f218,%f230,%f218,%f186
/*    166 */	fcmplted,s	%f74,%f88,%f88


/*    166 */	sxar2
/*    166 */	fcmplted,s	%f72,%f104,%f104
/*    166 */	fnmsubd,s	%f140,%f234,%f236,%f140


/*    166 */	sxar2
/*    166 */	fmuld,s	%f174,%f174,%f242
/*    166 */	fmuld,s	%f176,%f176,%f244


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f130,%f138,%f130,%f130
/*    166 */	fnmsubd,s	%f204,%f238,%f236,%f238


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f206,%f240,%f236,%f240
/*    166 */	fand,s	%f120,%f70,%f120


/*    166 */	sxar2
/*    166 */	fmuld,s	%f184,%f184,%f246
/*    166 */	fnmsubd,sc	%f324,%f118,%f110,%f118


/*    166 */	sxar2
/*    166 */	fmuld,s	%f186,%f186,%f248
/*    166 */	fcmplted,s	%f74,%f108,%f108


/*    166 */	sxar2
/*    166 */	fcmplted,s	%f72,%f160,%f160
/*    166 */	fmaddd,s	%f132,%f140,%f132,%f132


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f178,%f242,%f236,%f178
/*    166 */	fnmsubd,s	%f180,%f244,%f236,%f180


/*    166 */	sxar2
/*    166 */	fand,s	%f130,%f84,%f130
/*    166 */	fmaddd,s	%f200,%f238,%f200,%f200


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f202,%f240,%f202,%f202
/*    166 */	fnmsubd,sc	%f324,%f120,%f52,%f68


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f220,%f246,%f236,%f246
/*    166 */	fnmsubd,s	%f222,%f248,%f236,%f248


/*    166 */	sxar2
/*    166 */	fcmplted,s	%f74,%f164,%f164
/*    166 */	fcmplted,s	%f72,%f182,%f182


/*    166 */	sxar2
/*    166 */	fcmplted,s	%f74,%f190,%f190
/*    166 */	fand,s	%f132,%f88,%f132


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f174,%f178,%f174,%f174
/*    166 */	fmaddd,s	%f176,%f180,%f176,%f176


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f342,%f130,%f118,%f130
/*    166 */	fmuld,s	%f200,%f200,%f250


/*    166 */	sxar2
/*    166 */	fmuld,s	%f202,%f202,%f252
/*    166 */	fmaddd,s	%f184,%f246,%f184,%f184


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f186,%f248,%f186,%f186
/*    166 */	fnmsubd,sc	%f342,%f132,%f68,%f86


/*    166 */	sxar2
/*    166 */	fand,s	%f174,%f104,%f174
/*    166 */	fand,s	%f176,%f108,%f176


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f204,%f250,%f236,%f204
/*    166 */	fnmsubd,s	%f206,%f252,%f236,%f206


/*    166 */	sxar2
/*    166 */	fmuld,s	%f184,%f184,%f254
/*    166 */	fmuld,s	%f186,%f186,%f34


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f362,%f174,%f130,%f174
/*    166 */	fnmsubd,sc	%f362,%f176,%f86,%f106


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f200,%f204,%f200,%f200
/*    166 */	fmaddd,s	%f202,%f206,%f202,%f202


/*    166 */	sxar2
/*    166 */	fnmsubd,s	%f220,%f254,%f236,%f220
/*    166 */	fnmsubd,s	%f222,%f34,%f236,%f222


/*    166 */	sxar2
/*    166 */	fand,s	%f200,%f160,%f200
/*    166 */	fand,s	%f202,%f164,%f202


/*    166 */	sxar2
/*    166 */	fmaddd,s	%f184,%f220,%f184,%f184
/*    166 */	fmaddd,s	%f186,%f222,%f186,%f186


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f418,%f200,%f174,%f200
/*    166 */	fnmsubd,sc	%f418,%f202,%f106,%f162


/*    166 */	sxar2
/*    166 */	fand,s	%f184,%f182,%f184
/*    166 */	fand,s	%f186,%f190,%f186


/*    166 */	sxar2
/*    166 */	fnmsubd,sc	%f444,%f184,%f200,%f184
/*    166 */	fnmsubd,sc	%f444,%f186,%f162,%f186

.L4997:


.L4996:


.L4999:


/*    149 */	sxar2
/*    149 */	add	%xg12,%xg10,%xg25
/* #00006 */	ldd,s	[%fp+255],%f244


/*     22 */	sxar2
/*     22 */	add	%xg4,%xg10,%xg26
/*     22 */	ldd,s	[%xg25],%f132


/*    191 */	sxar2
/*    191 */	ldd,s	[%xg25+16],%f140
/*    191 */	add	%xg10,64,%xg10


/*     38 */	sxar2
/*     38 */	subcc	%xg8,2,%xg8
/* #00006 */	ldd,s	[%fp+239],%f246


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


/*    177 */	sxar2
/*    177 */	fnmsubd,sc	%f430,%f184,%f144,%f184
/*    177 */	fnmsubd,sc	%f430,%f186,%f140,%f186

/*    191 */	bpos,pt	%icc, .L4999
	nop


.L4995:


.L4889:

/*    178 */	sxar1
/*    178 */	addcc	%xg8,1,%xg8

/*    178 */	bneg	.L4882
	nop


.L4890:

/*    178 */	sxar1
/*    178 */	ldx	[%i0+2199],%xg28

.L4895:


/*    149 */	sxar2
/*    149 */	add	%xg28,%xg10,%xg27
/* #00005 */	ldd,s	[%fp+255],%f240


/*     22 */	sxar2
/*     22 */	add	%xg10,32,%xg10
/*     22 */	ldd,s	[%xg27],%f200


/*    191 */	sxar2
/*    191 */	ldd,s	[%xg27+16],%f208
/*    191 */	subcc	%xg8,1,%xg8


/*    149 */	sxar2
/* #00005 */	ldd,s	[%fp+239],%f242
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


/*    176 */	sxar2
/*    176 */	fand,s	%f220,%f210,%f220
/*    176 */	fnmsubd,sc	%f464,%f212,%f184,%f184

/*    177 */	sxar1
/*    177 */	fnmsubd,sc	%f464,%f220,%f186,%f186

/*    191 */	bpos,pt	%icc, .L4895
	nop


.L4891:


.L4882:


/*    191 */	sxar2
/*    191 */	std,s	%f184,[%fp+1263]
/*    191 */	std,s	%f186,[%fp+1279]

.L4883:


/*     22 */	sxar2
/*     22 */	add	%xg19,%xg1,%xg18
/*     22 */	ldd,s	[%fp+1263],%f236



/*    180 */	sxar2
/*    180 */	add	%xg1,32,%xg1
/*    180 */	add	%xg3,128,%xg3


/*     24 */	sxar2
/*     24 */	subcc	%xg0,1,%xg0
/*     24 */	std,s	%f236,[%xg18]


/*     25 */	sxar2
/*     25 */	ldd,s	[%fp+1279],%f238
/*     25 */	std,s	%f238,[%xg18+16]

/*    180 */	bne,pt	%icc, .L4877
/*    180 */	add	%o0,4,%o0


.L4884:

/*    180 */
/*    180 */	ba	.L4875
	nop


.L4886:

/*    180 *//*    180 */	call	__mpc_obar
/*    180 */	ldx	[%fp+2199],%o0

/*    180 *//*    180 */	call	__mpc_obar
/*    180 */	ldx	[%fp+2199],%o0


.L4887:

/*    180 */	ret
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
	.skip	786432
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
	.byte	50
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
.LB0..113.1:
	.skip	8
	.type	.LB0..113.1,#object
	.size	.LB0..113.1,.-.LB0..113.1
	.section	".rodata"
	.align	8
.LR0.cnt.5:
	.word	0XC0080000,0
	.type	.LR0.cnt.5,#object
	.size	.LR0.cnt.5,.-.LR0.cnt.5
	.section	".rodata"
	.align	8
.LR0.cnt.4:
	.word	0X3FE00000,0
	.type	.LR0.cnt.4,#object
	.size	.LR0.cnt.4,.-.LR0.cnt.4
	.section	".rodata"
	.align	8
.LR0.cnt.3:
	.word	0X3FF00000,0
	.type	.LR0.cnt.3,#object
	.size	.LR0.cnt.3,.-.LR0.cnt.3
	.section	".rodata"
	.align	8
.LR0.cnt.2:
	.word	0,0
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
.LS0.cnt.6:
	.word	1065353216
	.type	.LS0.cnt.6,#object
	.size	.LS0.cnt.6,.-.LS0.cnt.6
