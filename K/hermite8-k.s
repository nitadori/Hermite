	.ident	"$Options: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) --preinclude //opt/FJSVfxlang/1.2.1/bin/../lib/FCC.pre --g++ -D__FUJITSU -Dunix -Dsparc -D__sparc__ -D__unix -D__sparc -D__BUILTIN_VA_ARG_INCR -D_OPENMP=200805 -D__PRAGMA_REDEFINE_EXTNAME -D__FCC_VERSION=600 -D__USER_LABEL_PREFIX__= -D__OPTIMIZE__ -D__HPC_ACE__ -D__ELF__ -D__linux -Asystem(unix) -Dlinux -D__LIBC_6B -D_LP64 -D__LP64__ --K=omp -DEIGHTH -DHPC_ACE_GRAVITY -I/opt/FJSVfxlang/1.2.1/include/mpi/fujitsu --K=noocl -D_REENTRANT -D__MT__ --lp --zmode=64 --sys_include=//opt/FJSVfxlang/1.2.1/bin/../include/c++/std --sys_include=//opt/FJSVfxlang/1.2.1/bin/../include/c++ --sys_include=//opt/FJSVfxlang/1.2.1/bin/../include --sys_include=/opt/FJSVXosDevkit/sparc64fx/target/usr/include --K=opt -D__sparcv9 -D__sparc_v9__ -D__arch64__ --exceptions ../SRC/hermite8-k.cpp -- -ncmdname=FCCpx -Nnoline -Kdalign -zobe=no-static-clump -zobe=cplus -O3 -x- -KSPARC64IXfx,dalign,ns,mfunc,lib,eval,rdconv,prefetch_conditional,fp_contract,fp_relaxed,ilfunc,fast_matmul -Kthreadsafe -O3 -x- -KSPARC64IXfx,dalign,ns,mfunc,lib,eval,rdconv,prefetch_conditional,fp_contract,fp_relaxed,ilfunc,fast_matmul,uxsimd,optmsg=2 -x32 -Nsrc -Kopenmp,threadsafe -KLP -zsrc=../SRC/hermite8-k.cpp hermite8-k.s $"
	.file	"hermite8-k.cpp"
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite8-k.cpp _ZNKSt12_String_baseIcSt20__iostring_allocatorIcEE21_M_throw_length_errorEv $"
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
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite8-k.cpp _ZNSt12_String_baseIcSt20__iostring_allocatorIcEE17_M_allocate_blockEm $"
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


.L1109:

/*    637 */	ret
	restore



.L210:


/*     61 */	cmp	%i1,257

/*     61 */	bleu,pt	%xcc, .L1106
/*     61 */	add	%i0,16,%o0


.L1105:

/*    458 */	cmp	%i1,-1

/*    458 */	bgu,pn	%xcc, .L1078
	nop


.L1086:


/*    123 */	call	_Znwm
/*    123 */	mov	%i1,%o0


.L6712:

/*    123 */	cmp	%o0,%g0

/*    123 */	be,pt	%xcc, .L1089
	nop


.L1093:


.L1106:

/*    660 */	add	%o0,%i1,%i1

/*    657 */	stx	%o0,[%i0]

/*    658 */	stx	%o0,[%i0+8]

/*    660 */	stx	%i1,[%i0+280]

/*      0 */	ret
	restore



.L1089:


/*    123 */	call	__cxa_allocate_exception
/*    123 */	mov	8,%o0
/*    123 */	mov	%o0,%l1
/*    123 */	call	_ZNSt9bad_allocC1Ev
/*    123 */	mov	%l1,%o0


.L7124:

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


.L1078:


/*    459 */	call	__cxa_allocate_exception
/*    459 */	mov	8,%o0
/*    459 */	mov	%o0,%l0
/*    459 */	call	_ZNSt9bad_allocC1Ev
/*    459 */	mov	%l0,%o0


.L7123:

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
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite8-k.cpp _ZN7Gravity6GForceC1Ev $"
	.section	".text._ZN7Gravity6GForceC1Ev",#alloc,#execinstr

	.weak	_ZN7Gravity6GForceC1Ev
	.align	64
_ZN7Gravity6GForceC1Ev:
.LLFB3:
.L529:

/*     37 */

.L530:


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


/*     25 */	sxar2
/*     25 */	std,s	%f32,[%o0+144]
/*     25 */	std,s	%f32,[%o0+160]

/*     25 */	sxar1
/*     25 */	std,s	%f32,[%o0+176]

/*     25 */	retl
	nop



.L531:


.LLFE3:
	.size	_ZN7Gravity6GForceC1Ev,.-_ZN7Gravity6GForceC1Ev
	.type	_ZN7Gravity6GForceC1Ev,#function
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite8-k.cpp _ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE $"
	.section	".text"
	.global	_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE
	.align	64
_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE:
.LLFB4:
.L541:

/*      6 */	save	%sp,-336,%sp
.LLCFI2:
/*      6 */	stx	%i2,[%fp+2191]
/*      6 */	stx	%i3,[%fp+2199]

.L542:

/*     26 */	sxar1
/*     26 */	fmovd	%f2,%f258


/*     13 */	srl	%i0,31,%g1

/*     13 */	add	%g1,%i0,%g1

/*     13 */	sra	%g1,1,%g1

/*     13 */	stw	%g1,[%fp+2031]

/*     26 */	sxar1
/*     26 */	std,s	%f2,[%fp+1903]

/*     14 *//*     14 */	sethi	%h44(_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE._OMP_1),%o0
/*     14 */	mov	%fp,%o1
/*     14 */	or	%o0,%m44(_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE._OMP_1),%o0
/*     14 */	mov	%g0,%o2
/*     14 */	sllx	%o0,12,%o0
/*     14 */	call	__mpc_opar
/*     14 */	or	%o0,%l44(_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE._OMP_1),%o0

/*     59 */
/*     59 */	ret
	restore



.L568:


.LLFE4:
	.size	_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE,.-_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE
	.type	_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE,#function
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite8-k.cpp _ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE._OMP_1 $"
	.section	".text"
	.align	64
_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE._OMP_1:
.LLFB5:
.L6897:

/*     14 */	save	%sp,-960,%sp
.LLCFI3:
/*     14 */	stx	%i0,[%fp+2175]
/*     14 */	stx	%i3,[%fp+2199]
/*     14 */	stx	%i0,[%fp+2175]

.L6898:

/*     14 *//*     14 */	sxar1
/*     14 */	ldsw	[%i0+2035],%xg13
/*     14 */
/*     14 */
/*     14 */
/*     15 */	ldsw	[%i0+2031],%l0
/*     15 */	cmp	%l0,%g0
/*     15 */	ble	.L6913
/*     15 */	mov	%g0,%o0


.L6899:

/*     15 */	sethi	%h44(.LR0.cnt.7),%g1

/*     15 */	sethi	%h44(.LR0.cnt.2),%g2

/*     15 */	sethi	%h44(.LR0.cnt.8),%g3


/*     15 */	sxar2
/*     15 */	sethi	%h44(.LR0.cnt.4),%xg0
/*     15 */	sethi	%h44(.LR0.cnt.3),%xg1

/*     15 */	sxar1
/*     15 */	sethi	%h44(.LR0.cnt.1),%xg2

/*     15 */	or	%g1,%m44(.LR0.cnt.7),%g1

/*     15 */	or	%g2,%m44(.LR0.cnt.2),%g2

/*     15 */	or	%g3,%m44(.LR0.cnt.8),%g3


/*     15 */	sxar2
/*     15 */	or	%xg0,%m44(.LR0.cnt.4),%xg0
/*     15 */	or	%xg1,%m44(.LR0.cnt.3),%xg1

/*     15 */	sxar1
/*     15 */	or	%xg2,%m44(.LR0.cnt.1),%xg2

/*     15 */	sllx	%g1,12,%g1

/*     15 */	sllx	%g2,12,%g2

/*     15 */	sllx	%g3,12,%g3


/*     15 */	sxar2
/*     15 */	sllx	%xg0,12,%xg0
/*     15 */	sllx	%xg1,12,%xg1

/*     15 */	sxar1
/*     15 */	sllx	%xg2,12,%xg2

/*     15 */	or	%g1,%l44(.LR0.cnt.7),%g1

/*     15 */	or	%g2,%l44(.LR0.cnt.2),%g2

/*     15 */	or	%g3,%l44(.LR0.cnt.8),%g3

/*     15 */	ldd	[%g1],%f48


/*     15 */	sxar2
/*     15 */	or	%xg0,%l44(.LR0.cnt.4),%xg0
/*     15 */	or	%xg1,%l44(.LR0.cnt.3),%xg1


/*     15 */	sxar2
/*     15 */	ldd	[%g1],%f304
/*     15 */	or	%xg2,%l44(.LR0.cnt.1),%xg2


/*     15 */	sxar1
/*     15 */	mov	1,%xg12

/*     15 */	sra	%l0,%g0,%l0

/*     15 */	ldd	[%g2],%f50

/*     15 */	sxar1
/*     15 */	ldd	[%g2],%f306


/*     15 */	ldd	[%g3],%f52



/*     15 */	sxar2
/*     15 */	ldd	[%g3],%f308
/*     15 */	ldd	[%xg0],%f54



/*     15 */	sxar2
/*     15 */	ldd	[%xg0],%f310
/*     15 */	ldd	[%xg1],%f56



/*     15 */	sxar2
/*     15 */	ldd	[%xg1],%f312
/*     15 */	ldd	[%xg2],%f58



/*     15 */	sxar2
/*     15 */	ldd	[%xg2],%f314
/*     15 */	stx	%xg12,[%fp+2031]


/*     15 */	sxar2
/*    ??? */	std,s	%f48,[%fp+1391]
/*    ??? */	std,s	%f50,[%fp+1375]


/*     15 */	sxar2
/*    ??? */	std,s	%f52,[%fp+1359]
/*    ??? */	std,s	%f54,[%fp+1343]


/*     15 */	sxar2
/*    ??? */	std,s	%f56,[%fp+1327]
/*    ??? */	std,s	%f58,[%fp+1311]

.L6900:

/*     15 */	add	%fp,2039,%l1

/*     15 */	mov	1,%l5

/*     15 */	add	%fp,2023,%l2

/*     15 */	add	%fp,2031,%l3

/*     15 */	sra	%l5,%g0,%l4

.L6901:

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
/*     15 */	ble,pn	%xcc, .L6913
	nop


.L6902:

/*     15 */	ldx	[%fp+2039],%o0


/*     15 */	sxar2
/*     15 */	ldx	[%fp+2023],%xg0
/*     15 */	ldx	[%i0+2191],%xg4


/*     15 */	sxar2
/*     15 */	ldx	[%i0+2199],%xg5
/*     15 */	ldd,s	[%i0+1903],%f84

/*     15 */	sra	%o0,%g0,%o0


/*     15 */	sxar2
/*     15 */	sra	%xg0,%g0,%xg0
/*     15 */	sub	%xg0,%o0,%xg0


/*     15 */	sxar2
/*     15 */	sra	%o0,%g0,%xg1
/*     15 */	add	%xg0,1,%xg0


/*     15 */	sxar2
/*     15 */	mulx	%xg1,416,%xg2
/*     15 */	sra	%xg0,%g0,%xg0


/*     15 */	sxar2
/*     15 */	sub	%l4,%xg0,%xg0
/*     15 */	srax	%xg0,32,%xg3


/*     15 */	sxar2
/*     15 */	and	%xg0,%xg3,%xg0
/*     15 */	sub	%l5,%xg0,%xg0


/*     15 */	sxar2
/*     15 */	cmp	%xg0,10
/*     15 */	mulx	%xg1,208,%xg1


/*     15 */	sxar2
/*     15 */	add	%xg4,%xg2,%xg4
/*     15 */	add	%xg5,%xg1,%xg5

/*     15 */	bl	.L7075
	nop


.L7071:


.L7079:


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+112],%f44
/*     15 */	add	%xg4,416,%xg6


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+528],%f48
/*     15 */	add	%xg4,832,%xg7


/*     15 */	sxar2
/*     15 */	add	%xg4,1248,%xg8
/*    ??? */	ldd,s	[%fp+1343],%f188


/*     15 */	sxar2
/*    ??? */	ldd,s	[%fp+1311],%f182
/*     15 */	ldd,s	[%xg4+368],%f68


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+320],%f142
/*     15 */	ldd,s	[%xg4+400],%f150


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+352],%f148
/*     15 */	fsubd,s	%f84,%f44,%f44


/*     15 */	sxar2
/*     15 */	fsubd,s	%f84,%f48,%f48
/*    ??? */	ldd,s	[%fp+1359],%f184


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+272],%f160
/*    ??? */	ldd,s	[%fp+1327],%f178


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+304],%f88
/*     15 */	ldd,s	[%xg4+944],%f52


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+784],%f146
/*     15 */	ldd,s	[%xg4+736],%f144


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+224],%f172
/*    ??? */	ldd,s	[%fp+1375],%f186


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+256],%f120
/*     15 */	ldd,s	[%xg4+688],%f108


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+384],%f154
/*     15 */	fmuld,s	%f188,%f44,%f128


/*     15 */	sxar2
/*     15 */	fmuld,s	%f182,%f44,%f130
/*     15 */	ldd,s	[%xg4+336],%f152


/*     15 */	sxar2
/*     15 */	fmuld,s	%f184,%f44,%f132
/*     15 */	fmuld,s	%f188,%f48,%f134


/*     15 */	sxar2
/*     15 */	fsubd,s	%f84,%f52,%f52
/*     15 */	fmuld,s	%f178,%f44,%f136


/*     15 */	sxar2
/*    ??? */	ldd,s	[%fp+1391],%f180
/*     15 */	ldd,s	[%xg4+640],%f92


/*     15 */	sxar2
/*     15 */	fmuld,s	%f186,%f44,%f122
/*     15 */	fmuld,s	%f184,%f48,%f124


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+176],%f176
/*     15 */	fmuld,s	%f186,%f48,%f106


/*     15 */	sxar2
/*     15 */	fmuld,s	%f178,%f48,%f140
/*     15 */	ldd,s	[%xg4+288],%f162


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+208],%f102
/*     15 */	fmuld,s	%f182,%f48,%f158


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+1360],%f64
/*     15 */	fmaddd,s	%f128,%f68,%f142,%f100


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f130,%f68,%f142,%f118
/*     15 */	ldd,s	[%xg4+1200],%f170


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f134,%f146,%f144,%f36
/*     15 */	fmaddd,s	%f128,%f150,%f148,%f94


/*     15 */	sxar2
/*     15 */	fmuld,s	%f180,%f44,%f46
/*     15 */	ldd,s	[%xg4+1152],%f166


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f132,%f68,%f142,%f96
/*     15 */	fmuld,s	%f188,%f52,%f138


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+816],%f82
/*     15 */	fmaddd,s	%f136,%f150,%f148,%f126


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+768],%f80
/*     15 */	fmaddd,s	%f132,%f154,%f152,%f42


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f136,%f68,%f142,%f68
/*     15 */	fmuld,s	%f180,%f48,%f50


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+128],%f104
/*     15 */	fmaddd,s	%f130,%f154,%f152,%f156


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+240],%f174
/*     15 */	fmaddd,s	%f132,%f100,%f160,%f100


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f136,%f118,%f160,%f118
/*     15 */	ldd,s	[%xg4+1104],%f72


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f124,%f36,%f108,%f36
/*     15 */	fmaddd,s	%f132,%f94,%f88,%f94


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+160],%f90
/*     15 */	fmuld,s	%f184,%f52,%f74


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f122,%f96,%f160,%f96
/*     15 */	fsubd,s	%f84,%f64,%f64


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+592],%f86
/*     15 */	fmaddd,s	%f128,%f126,%f88,%f126


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+720],%f114
/*     15 */	fmaddd,s	%f122,%f42,%f162,%f42


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f128,%f68,%f160,%f68
/*     15 */	ldd,s	[%xg4+672],%f98


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f136,%f156,%f162,%f156
/*     15 */	fmaddd,s	%f136,%f154,%f152,%f164


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+64],%f116
/*     15 */	fmaddd,s	%f122,%f100,%f172,%f100


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f128,%f118,%f172,%f118
/*     15 */	ldd,s	[%xg4+1056],%f110


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f130,%f150,%f148,%f130
/*     15 */	fmaddd,s	%f122,%f94,%f120,%f94


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+192],%f40
/*     15 */	fmaddd,s	%f138,%f170,%f166,%f168


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f106,%f36,%f92,%f36
/*     15 */	ldd,s	[%xg4+1232],%f78


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f132,%f126,%f120,%f126
/*     15 */	fmaddd,s	%f46,%f96,%f172,%f96


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+1184],%f76
/*     15 */	fmaddd,s	%f46,%f42,%f174,%f42


/*     15 */	sxar2
/*     15 */	fmuld,s	%f186,%f52,%f112
/*     15 */	fmuld,s	%f188,%f64,%f32


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f158,%f146,%f144,%f190
/*     15 */	fmaddd,s	%f46,%f100,%f176,%f100


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f132,%f118,%f176,%f118
/*     15 */	fmaddd,s	%f132,%f68,%f172,%f68


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f46,%f94,%f102,%f94
/*     15 */	fmaddd,s	%f128,%f156,%f174,%f156


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f50,%f36,%f86,%f36
/*     15 */	fmaddd,s	%f44,%f96,%f176,%f96


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f128,%f164,%f162,%f164
/*     15 */	fmaddd,s	%f136,%f130,%f88,%f136


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f140,%f82,%f80,%f192
/*     15 */	fmaddd,s	%f74,%f168,%f72,%f168


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f128,%f154,%f152,%f154
/*     15 */	fmaddd,s	%f134,%f82,%f80,%f194


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f44,%f100,%f104,%f100
/*     15 */	fmaddd,s	%f122,%f118,%f104,%f118

/*     15 */	sxar1
/*     15 */	fmaddd,s	%f44,%f94,%f90,%f94

.L6903:


/*     15 */	sxar2
/*     15 */	fmuld,s	%f178,%f52,%f34
/*     15 */	fmaddd,s	%f124,%f146,%f144,%f38


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg6+384],%f58
/*     15 */	ldd,s	[%xg6+336],%f54


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f122,%f68,%f176,%f68
/*     15 */	fmaddd,s	%f132,%f156,%f40,%f156


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+144],%f60
/*     15 */	fmaddd,s	%f132,%f164,%f174,%f164


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f128,%f136,%f120,%f128
/*     15 */	fmaddd,s	%f46,%f118,%f116,%f118


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f134,%f192,%f114,%f192
/*     15 */	ldd,s	[%xg4],%f62


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f112,%f168,%f110,%f168
/*     15 */	fmaddd,s	%f132,%f154,%f162,%f154


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+304],%f196
/*     15 */	fmaddd,s	%f140,%f190,%f108,%f190


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f124,%f194,%f114,%f194
/*     15 */	fmaddd,s	%f106,%f38,%f108,%f38


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f124,%f58,%f54,%f56
/*     15 */	ldd,s	[%xg6+288],%f66


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f46,%f68,%f104,%f68
/*     15 */	fmaddd,s	%f122,%f156,%f60,%f156


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg6+128],%f104
/*     15 */	ldd,s	[%xg4+80],%f70


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f122,%f164,%f40,%f164
/*     15 */	fmaddd,s	%f132,%f128,%f102,%f128


/*     15 */	sxar2
/*     15 */	std,s	%f100,[%xg5+112]
/*     15 */	fmaddd,s	%f44,%f118,%f62,%f118


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f124,%f192,%f98,%f192
/*     15 */	std,s	%f96,[%xg5+160]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f122,%f126,%f102,%f126
/*     15 */	fmaddd,s	%f122,%f154,%f174,%f154


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+256],%f214
/*     15 */	ldd,s	[%xg6+208],%f130


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f106,%f194,%f98,%f194
/*     15 */	fmaddd,s	%f132,%f150,%f148,%f132


/*     15 */	sxar2
/*     15 */	std,s	%f94,[%xg5+144]
/*     15 */	fmaddd,s	%f50,%f38,%f92,%f38


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f106,%f56,%f66,%f56
/*     15 */	ldd,s	[%xg6+240],%f136


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f44,%f68,%f116,%f68
/*     15 */	fmaddd,s	%f46,%f156,%f70,%f156


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+16],%f148
/*     15 */	fmaddd,s	%f46,%f164,%f60,%f164


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f122,%f128,%f90,%f128
/*     15 */	fmaddd,s	%f140,%f146,%f144,%f146


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+96],%f150
/*     15 */	fmaddd,s	%f48,%f36,%f104,%f36


/*     15 */	sxar2
/*     15 */	fmuld,s	%f180,%f52,%f116
/*     15 */	fmaddd,s	%f46,%f154,%f40,%f154


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+528],%f172
/*     15 */	ldd,s	[%xg8+368],%f204


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f50,%f194,%f130,%f194
/*     15 */	fmaddd,s	%f122,%f132,%f88,%f122


/*     15 */	sxar2
/*     15 */	std,s	%f118,[%xg5]
/*     15 */	fmaddd,s	%f48,%f38,%f86,%f38


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f50,%f56,%f136,%f56
/*     15 */	std,s	%f68,[%xg5+64]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f158,%f58,%f54,%f142
/*     15 */	fmaddd,s	%f44,%f156,%f148,%f156


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+272],%f218
/*     15 */	fmaddd,s	%f44,%f164,%f70,%f164


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f46,%f128,%f150,%f128
/*     15 */	ldd,s	[%xg6+64],%f228


/*     15 */	sxar2
/*     15 */	fmuld,s	%f182,%f52,%f152
/*     15 */	ldd,s	[%xg4+32],%f176


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f46,%f126,%f90,%f126
/*     15 */	ldd,s	[%xg8+320],%f200


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f134,%f190,%f92,%f190
/*     15 */	ldd,s	[%xg6+160],%f198


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f44,%f154,%f60,%f154
/*     15 */	fmaddd,s	%f46,%f122,%f120,%f46


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f44,%f42,%f40,%f42
/*     15 */	ldd,s	[%xg8+224],%f230


/*     15 */	sxar2
/*     15 */	fmuld,s	%f184,%f64,%f160
/*     15 */	fsubd,s	%f84,%f172,%f172


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f134,%f146,%f108,%f146
/*     15 */	fmaddd,s	%f140,%f142,%f66,%f142


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+176],%f210
/*     15 */	std,s	%f156,[%xg5+16]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f140,%f58,%f54,%f174
/*     15 */	fmaddd,s	%f44,%f128,%f176,%f128


/*     15 */	sxar2
/*     15 */	std,s	%f164,[%xg5+80]
/*     15 */	fmaddd,s	%f158,%f82,%f80,%f158


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f44,%f126,%f150,%f126
/*     15 */	ldd,s	[%xg6+192],%f226


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f124,%f190,%f86,%f190
/*     15 */	ldd,s	[%xg8+400],%f242


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f48,%f194,%f198,%f194
/*     15 */	std,s	%f154,[%xg5+128]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f32,%f204,%f200,%f202
/*     15 */	fmaddd,s	%f44,%f46,%f102,%f44


/*     15 */	sxar2
/*     15 */	std,s	%f42,[%xg5+176]
/*     15 */	fmuld,s	%f186,%f64,%f206


/*     15 */	sxar2
/*     15 */	fmuld,s	%f188,%f172,%f208
/*     15 */	fmaddd,s	%f116,%f168,%f210,%f168


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f152,%f170,%f166,%f212
/*     15 */	ldd,s	[%xg4+48],%f176


/*     15 */	sxar2
/*     15 */	std,s	%f128,[%xg5+32]
/*     15 */	fmaddd,s	%f124,%f146,%f92,%f146


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f134,%f142,%f136,%f142
/*     15 */	std,s	%f126,[%xg5+96]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f134,%f174,%f66,%f174
/*     15 */	fmaddd,s	%f140,%f158,%f114,%f140


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f106,%f190,%f104,%f190
/*     15 */	fmaddd,s	%f34,%f78,%f76,%f216


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+352],%f254
/*     15 */	std,s	%f44,[%xg5+192]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f160,%f202,%f218,%f202
/*     15 */	fmaddd,s	%f134,%f58,%f54,%f58


/*     15 */	sxar2
/*     15 */	std,s	%f176,[%xg5+48]
/*     15 */	fmaddd,s	%f138,%f78,%f76,%f220


/*     15 */	sxar2
/*     15 */	fmuld,s	%f178,%f64,%f222
/*     15 */	fmaddd,s	%f74,%f170,%f166,%f224


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+384],%f236
/*     15 */	ldd,s	[%xg7+336],%f232


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f106,%f146,%f86,%f146
/*     15 */	fmaddd,s	%f124,%f142,%f226,%f142


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg6+144],%f238
/*     15 */	fmaddd,s	%f124,%f174,%f136,%f174


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f134,%f140,%f98,%f134
/*     15 */	fmaddd,s	%f50,%f190,%f228,%f190


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f138,%f216,%f196,%f216
/*     15 */	ldd,s	[%xg6],%f240


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f206,%f202,%f230,%f202
/*     15 */	fmaddd,s	%f124,%f58,%f66,%f58


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+304],%f54
/*     15 */	fmaddd,s	%f34,%f212,%f72,%f212


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f74,%f220,%f196,%f220
/*     15 */	fmaddd,s	%f112,%f224,%f72,%f224


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f74,%f236,%f232,%f234
/*     15 */	ldd,s	[%xg7+288],%f244


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f50,%f146,%f104,%f146
/*     15 */	fmaddd,s	%f106,%f142,%f238,%f142


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+128],%f248
/*     15 */	ldd,s	[%xg6+80],%f246


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f106,%f174,%f226,%f174
/*     15 */	fmaddd,s	%f124,%f134,%f130,%f134


/*     15 */	sxar2
/*     15 */	std,s	%f36,[%xg5+320]
/*     15 */	fmaddd,s	%f48,%f190,%f240,%f190


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f74,%f216,%f214,%f216
/*     15 */	std,s	%f38,[%xg5+368]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f106,%f192,%f130,%f192
/*     15 */	fmaddd,s	%f106,%f58,%f136,%f58


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+256],%f86
/*     15 */	ldd,s	[%xg7+208],%f252


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f112,%f220,%f214,%f220
/*     15 */	fmaddd,s	%f124,%f82,%f80,%f124


/*     15 */	sxar2
/*     15 */	std,s	%f194,[%xg5+352]
/*     15 */	fmaddd,s	%f116,%f224,%f110,%f224


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f112,%f234,%f244,%f234
/*     15 */	ldd,s	[%xg7+240],%f36


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f48,%f146,%f228,%f146
/*     15 */	fmaddd,s	%f50,%f142,%f246,%f142


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg6+16],%f40
/*     15 */	fmaddd,s	%f50,%f174,%f238,%f174


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f106,%f134,%f198,%f134
/*     15 */	fmaddd,s	%f34,%f170,%f166,%f170


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg6+96],%f42
/*     15 */	fmaddd,s	%f52,%f168,%f248,%f168


/*     15 */	sxar2
/*     15 */	add	%xg8,832,%xg4
/*     15 */	fmuld,s	%f180,%f64,%f250


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f50,%f58,%f226,%f58
/*     15 */	ldd,s	[%xg8+944],%f44


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+784],%f66
/*     15 */	fmaddd,s	%f116,%f220,%f252,%f220


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f106,%f124,%f114,%f106
/*     15 */	std,s	%f190,[%xg5+208]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f52,%f224,%f210,%f224
/*     15 */	fmaddd,s	%f116,%f234,%f36,%f234


/*     15 */	sxar2
/*     15 */	std,s	%f146,[%xg5+272]
/*     15 */	fmaddd,s	%f152,%f236,%f232,%f38


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f48,%f142,%f40,%f142
/*     15 */	ldd,s	[%xg8+688],%f154


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f48,%f174,%f246,%f174
/*     15 */	fmaddd,s	%f50,%f134,%f42,%f134


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+64],%f166
/*     15 */	fmuld,s	%f182,%f64,%f46


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg6+32],%f68
/*     15 */	fmaddd,s	%f50,%f192,%f198,%f192


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+736],%f90
/*     15 */	fmaddd,s	%f138,%f212,%f110,%f212


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+160],%f88
/*     15 */	fmaddd,s	%f48,%f58,%f238,%f58


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f50,%f106,%f98,%f50
/*     15 */	fmaddd,s	%f48,%f56,%f226,%f56


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+640],%f190
/*     15 */	fmuld,s	%f184,%f172,%f60


/*     15 */	sxar2
/*     15 */	fsubd,s	%f84,%f44,%f44
/*     15 */	fmaddd,s	%f138,%f170,%f72,%f170


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f34,%f38,%f244,%f38
/*     15 */	ldd,s	[%xg8+176],%f140


/*     15 */	sxar2
/*     15 */	std,s	%f142,[%xg5+224]
/*     15 */	fmaddd,s	%f34,%f236,%f232,%f62


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f48,%f134,%f68,%f134
/*     15 */	std,s	%f174,[%xg5+288]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f152,%f78,%f76,%f152
/*     15 */	fmaddd,s	%f48,%f192,%f42,%f192


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+192],%f164
/*     15 */	fmaddd,s	%f74,%f212,%f210,%f212


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+816],%f194
/*     15 */	fmaddd,s	%f52,%f220,%f88,%f220


/*     15 */	sxar2
/*     15 */	std,s	%f58,[%xg5+336]
/*     15 */	fmaddd,s	%f208,%f66,%f90,%f98


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f48,%f50,%f130,%f48
/*     15 */	std,s	%f56,[%xg5+384]


/*     15 */	sxar2
/*     15 */	fmuld,s	%f186,%f172,%f130
/*     15 */	fmuld,s	%f188,%f44,%f128


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f250,%f202,%f140,%f202
/*     15 */	fmaddd,s	%f46,%f204,%f200,%f142


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg6+48],%f198
/*     15 */	std,s	%f134,[%xg5+240]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f74,%f170,%f110,%f170
/*     15 */	fmaddd,s	%f138,%f38,%f36,%f38


/*     15 */	sxar2
/*     15 */	std,s	%f192,[%xg5+304]
/*     15 */	fmaddd,s	%f138,%f62,%f244,%f62


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f34,%f152,%f196,%f34
/*     15 */	fmaddd,s	%f112,%f212,%f248,%f212


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f222,%f242,%f254,%f152
/*     15 */	ldd,s	[%xg8+768],%f238


/*     15 */	sxar2
/*     15 */	std,s	%f48,[%xg5+400]
/*     15 */	fmaddd,s	%f60,%f98,%f154,%f98


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f138,%f236,%f232,%f236
/*     15 */	std,s	%f198,[%xg5+256]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f32,%f242,%f254,%f156
/*     15 */	fmuld,s	%f178,%f172,%f158


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f160,%f204,%f200,%f162
/*     15 */	ldd,s	[%xg8+384],%f226


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+336],%f198
/*     15 */	fmaddd,s	%f112,%f170,%f210,%f170


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f74,%f38,%f164,%f38
/*     15 */	ldd,s	[%xg7+144],%f228


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f74,%f62,%f36,%f62
/*     15 */	fmaddd,s	%f138,%f34,%f214,%f138


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f116,%f212,%f166,%f212
/*     15 */	fmaddd,s	%f32,%f152,%f54,%f152


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7],%f240
/*     15 */	fmaddd,s	%f130,%f98,%f190,%f98


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f74,%f236,%f244,%f236
/*     15 */	ldd,s	[%xg8+720],%f244


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f222,%f142,%f218,%f142
/*     15 */	fmaddd,s	%f160,%f156,%f54,%f156


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f206,%f162,%f218,%f162
/*     15 */	fmaddd,s	%f160,%f226,%f198,%f210


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+288],%f246
/*     15 */	fmaddd,s	%f116,%f170,%f248,%f170


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f112,%f38,%f228,%f38
/*     15 */	ldd,s	[%xg8+128],%f34


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+80],%f248
/*     15 */	fmaddd,s	%f112,%f62,%f164,%f62


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f74,%f138,%f252,%f138
/*     15 */	std,s	%f168,[%xg5+528]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f52,%f212,%f240,%f212
/*     15 */	fmaddd,s	%f160,%f152,%f86,%f152


/*     15 */	sxar2
/*     15 */	std,s	%f224,[%xg5+576]
/*     15 */	fmaddd,s	%f112,%f216,%f252,%f216


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f112,%f236,%f36,%f236
/*     15 */	ldd,s	[%xg8+672],%f40


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+208],%f58
/*     15 */	fmaddd,s	%f206,%f156,%f86,%f156


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f74,%f78,%f76,%f74
/*     15 */	std,s	%f220,[%xg5+560]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f250,%f162,%f230,%f162
/*     15 */	fmaddd,s	%f206,%f210,%f246,%f210


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+240],%f72
/*     15 */	fmaddd,s	%f52,%f170,%f166,%f170


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f116,%f38,%f248,%f38
/*     15 */	ldd,s	[%xg7+16],%f76


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f116,%f62,%f228,%f62
/*     15 */	fmaddd,s	%f112,%f138,%f88,%f138


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f222,%f204,%f200,%f204
/*     15 */	ldd,s	[%xg7+96],%f78


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f64,%f202,%f34,%f202
/*     15 */	add	%xg8,1248,%xg6


/*     15 */	sxar2
/*     15 */	fmuld,s	%f180,%f172,%f56
/*     15 */	fmaddd,s	%f116,%f236,%f164,%f236


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+1360],%f48
/*     15 */	ldd,s	[%xg8+1200],%f68


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f250,%f156,%f58,%f156
/*     15 */	fmaddd,s	%f112,%f74,%f196,%f112


/*     15 */	sxar2
/*     15 */	std,s	%f212,[%xg5+416]
/*     15 */	fmaddd,s	%f64,%f162,%f140,%f162


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f250,%f210,%f72,%f210
/*     15 */	std,s	%f170,[%xg5+480]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f46,%f226,%f198,%f74
/*     15 */	fmaddd,s	%f52,%f38,%f76,%f38


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+1104],%f168
/*     15 */	fmaddd,s	%f52,%f62,%f248,%f62


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f116,%f138,%f78,%f138
/*     15 */	ldd,s	[%xg8+64],%f224


/*     15 */	sxar2
/*     15 */	fmuld,s	%f182,%f172,%f80
/*     15 */	ldd,s	[%xg7+32],%f174


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f116,%f216,%f88,%f216
/*     15 */	ldd,s	[%xg8+1152],%f70


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f32,%f142,%f230,%f142
/*     15 */	ldd,s	[%xg8+160],%f196


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f52,%f236,%f228,%f236
/*     15 */	fmaddd,s	%f116,%f112,%f214,%f116


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f52,%f234,%f164,%f234
/*     15 */	ldd,s	[%xg8+1056],%f192


/*     15 */	sxar2
/*     15 */	fmuld,s	%f184,%f44,%f132
/*     15 */	fsubd,s	%f84,%f48,%f48


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f32,%f204,%f218,%f204
/*     15 */	fmaddd,s	%f222,%f74,%f246,%f74


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+592],%f200
/*     15 */	std,s	%f38,[%xg5+432]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f222,%f226,%f198,%f170
/*     15 */	fmaddd,s	%f52,%f138,%f174,%f138


/*     15 */	sxar2
/*     15 */	std,s	%f62,[%xg5+496]
/*     15 */	fmaddd,s	%f46,%f242,%f254,%f46


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f52,%f216,%f78,%f216
/*     15 */	ldd,s	[%xg8+192],%f220


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f160,%f142,%f140,%f142
/*     15 */	ldd,s	[%xg8+1232],%f150


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f64,%f156,%f196,%f156
/*     15 */	std,s	%f236,[%xg5+544]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f128,%f68,%f70,%f100
/*     15 */	fmaddd,s	%f52,%f116,%f252,%f52


/*     15 */	sxar2
/*     15 */	std,s	%f234,[%xg5+592]
/*     15 */	fmuld,s	%f186,%f44,%f122


/*     15 */	sxar2
/*     15 */	fmuld,s	%f188,%f48,%f134
/*     15 */	fmaddd,s	%f56,%f98,%f200,%f98


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f80,%f66,%f90,%f212
/*     15 */	ldd,s	[%xg7+48],%f218


/*     15 */	sxar2
/*     15 */	std,s	%f138,[%xg5+448]
/*     15 */	fmaddd,s	%f160,%f204,%f230,%f204


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f32,%f74,%f72,%f74
/*     15 */	std,s	%f216,[%xg5+512]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f32,%f170,%f246,%f170
/*     15 */	fmaddd,s	%f222,%f46,%f54,%f222


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f206,%f142,%f34,%f142
/*     15 */	fmaddd,s	%f158,%f194,%f238,%f214


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+1184],%f148
/*     15 */	std,s	%f52,[%xg5+608]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f132,%f100,%f168,%f100
/*     15 */	fmaddd,s	%f32,%f226,%f198,%f226


/*     15 */	sxar2
/*     15 */	std,s	%f218,[%xg5+464]
/*     15 */	fmaddd,s	%f208,%f194,%f238,%f216


/*     15 */	sxar2
/*     15 */	fmuld,s	%f178,%f44,%f136
/*     15 */	fmaddd,s	%f60,%f66,%f90,%f218


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+800],%f232
/*     15 */	ldd,s	[%xg8+752],%f228


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f206,%f204,%f140,%f204
/*     15 */	fmaddd,s	%f160,%f74,%f220,%f74


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+144],%f234
/*     15 */	fmaddd,s	%f160,%f170,%f72,%f170


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f32,%f222,%f86,%f32
/*     15 */	fmaddd,s	%f250,%f142,%f224,%f142


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f208,%f214,%f244,%f214
/*     15 */	ldd,s	[%xg8],%f236


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f122,%f100,%f192,%f100
/*     15 */	fmaddd,s	%f160,%f226,%f246,%f226


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+1136],%f88
/*     15 */	fmaddd,s	%f158,%f212,%f154,%f212


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f60,%f216,%f244,%f216
/*     15 */	fmaddd,s	%f130,%f218,%f154,%f218


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f60,%f232,%f228,%f230
/*     15 */	ldd,s	[%xg8+704],%f240


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f250,%f204,%f34,%f204
/*     15 */	fmaddd,s	%f206,%f74,%f234,%f74


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+544],%f248
/*     15 */	ldd,s	[%xg8+80],%f246


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f206,%f170,%f220,%f170
/*     15 */	fmaddd,s	%f160,%f32,%f58,%f32


/*     15 */	sxar2
/*     15 */	std,s	%f202,[%xg5+736]
/*     15 */	fmaddd,s	%f64,%f142,%f236,%f142


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f60,%f214,%f40,%f214
/*     15 */	std,s	%f162,[%xg5+784]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f206,%f152,%f58,%f152
/*     15 */	fmaddd,s	%f206,%f226,%f72,%f226


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+1088],%f120
/*     15 */	ldd,s	[%xg8+624],%f252


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f130,%f216,%f40,%f216
/*     15 */	fmaddd,s	%f160,%f242,%f254,%f160


/*     15 */	sxar2
/*     15 */	std,s	%f156,[%xg5+768]
/*     15 */	fmaddd,s	%f56,%f218,%f190,%f218


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f130,%f230,%f240,%f230
/*     15 */	ldd,s	[%xg8+656],%f254


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f64,%f204,%f224,%f204
/*     15 */	fmaddd,s	%f250,%f74,%f246,%f74


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+16],%f36
/*     15 */	fmaddd,s	%f250,%f170,%f234,%f170


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f206,%f32,%f196,%f32
/*     15 */	fmaddd,s	%f158,%f66,%f90,%f66


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+96],%f38
/*     15 */	fmaddd,s	%f172,%f98,%f248,%f98


/*     15 */	sxar2
/*     15 */	add	%xg8,1664,%xg7
/*     15 */	fmuld,s	%f180,%f44,%f46


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f250,%f226,%f220,%f226
/*     15 */	ldd,s	[%xg8+1776],%f52


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+1616],%f146
/*     15 */	fmaddd,s	%f56,%f216,%f252,%f216


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f206,%f160,%f54,%f206
/*     15 */	std,s	%f142,[%xg5+624]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f172,%f218,%f200,%f218
/*     15 */	fmaddd,s	%f56,%f230,%f254,%f230


/*     15 */	sxar2
/*     15 */	std,s	%f204,[%xg5+688]
/*     15 */	fmaddd,s	%f80,%f232,%f228,%f34


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f64,%f74,%f36,%f74
/*     15 */	ldd,s	[%xg8+1520],%f108


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f64,%f170,%f246,%f170
/*     15 */	fmaddd,s	%f250,%f32,%f38,%f32


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+480],%f90
/*     15 */	fmuld,s	%f182,%f44,%f54


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+32],%f72
/*     15 */	fmaddd,s	%f250,%f152,%f196,%f152


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+1568],%f144
/*     15 */	fmaddd,s	%f208,%f212,%f190,%f212


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+576],%f76
/*     15 */	fmaddd,s	%f64,%f226,%f234,%f226


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f250,%f206,%f86,%f250
/*     15 */	fmaddd,s	%f64,%f210,%f220,%f210


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+1472],%f92
/*     15 */	fmuld,s	%f184,%f48,%f124


/*     15 */	sxar2
/*     15 */	fsubd,s	%f84,%f52,%f52
/*     15 */	fmaddd,s	%f208,%f66,%f154,%f66


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f158,%f34,%f240,%f34
/*     15 */	ldd,s	[%xg8+1008],%f176


/*     15 */	sxar2
/*     15 */	std,s	%f74,[%xg5+640]
/*     15 */	fmaddd,s	%f158,%f232,%f228,%f62


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f64,%f32,%f72,%f32
/*     15 */	std,s	%f170,[%xg5+704]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f80,%f194,%f238,%f80
/*     15 */	fmaddd,s	%f64,%f152,%f38,%f152


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+608],%f78
/*     15 */	fmaddd,s	%f60,%f212,%f200,%f212


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+1648],%f82
/*     15 */	fmaddd,s	%f172,%f216,%f76,%f216


/*     15 */	sxar2
/*     15 */	std,s	%f226,[%xg5+752]
/*     15 */	fmaddd,s	%f134,%f146,%f144,%f36


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f64,%f250,%f58,%f64
/*     15 */	std,s	%f210,[%xg5+800]


/*     15 */	sxar2
/*     15 */	fmuld,s	%f186,%f48,%f106
/*     15 */	fmuld,s	%f188,%f52,%f138


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f46,%f100,%f176,%f100
/*     15 */	fmaddd,s	%f54,%f68,%f70,%f118


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+48],%f220
/*     15 */	std,s	%f32,[%xg5+656]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f60,%f66,%f190,%f66
/*     15 */	fmaddd,s	%f208,%f34,%f254,%f34


/*     15 */	sxar2
/*     15 */	std,s	%f152,[%xg5+720]
/*     15 */	fmaddd,s	%f208,%f62,%f240,%f62


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f158,%f80,%f244,%f158
/*     15 */	fmaddd,s	%f130,%f212,%f248,%f212


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f136,%f150,%f148,%f126
/*     15 */	ldd,s	[%xg8+1600],%f80


/*     15 */	sxar2
/*     15 */	std,s	%f64,[%xg5+816]
/*     15 */	fmaddd,s	%f124,%f36,%f108,%f36


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f208,%f232,%f228,%f232
/*     15 */	std,s	%f220,[%xg5+672]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f128,%f150,%f148,%f94
/*     15 */	fmuld,s	%f178,%f48,%f140


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f132,%f68,%f70,%f96
/*     15 */	ldd,s	[%xg8+1216],%f154


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+1168],%f142
/*     15 */	fmaddd,s	%f130,%f66,%f200,%f66


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f60,%f34,%f78,%f34
/*     15 */	ldd,s	[%xg8+560],%f152


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f60,%f62,%f254,%f62
/*     15 */	fmaddd,s	%f208,%f158,%f40,%f208


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f56,%f212,%f90,%f212
/*     15 */	fmaddd,s	%f128,%f126,%f88,%f126


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+416],%f156
/*     15 */	fmaddd,s	%f106,%f36,%f92,%f36


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f60,%f232,%f240,%f232
/*     15 */	ldd,s	[%xg8+1552],%f114


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f136,%f118,%f168,%f118
/*     15 */	fmaddd,s	%f132,%f94,%f88,%f94


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f122,%f96,%f168,%f96
/*     15 */	fmaddd,s	%f132,%f154,%f142,%f42


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+1120],%f162
/*     15 */	fmaddd,s	%f56,%f66,%f248,%f66


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f130,%f34,%f152,%f34
/*     15 */	ldd,s	[%xg8+960],%f104


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+496],%f158
/*     15 */	fmaddd,s	%f130,%f62,%f78,%f62


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f60,%f208,%f252,%f208
/*     15 */	std,s	%f98,[%xg5+944]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f172,%f212,%f156,%f212
/*     15 */	fmaddd,s	%f132,%f126,%f120,%f126


/*     15 */	sxar2
/*     15 */	std,s	%f218,[%xg5+992]
/*     15 */	fmaddd,s	%f130,%f214,%f252,%f214


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f130,%f232,%f254,%f232
/*     15 */	ldd,s	[%xg8+1504],%f98


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+1040],%f102
/*     15 */	fmaddd,s	%f122,%f94,%f120,%f94


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f60,%f194,%f238,%f60
/*     15 */	std,s	%f216,[%xg5+976]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f46,%f96,%f192,%f96
/*     15 */	fmaddd,s	%f122,%f42,%f162,%f42


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+1072],%f174
/*     15 */	fmaddd,s	%f172,%f66,%f90,%f66


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f56,%f34,%f158,%f34
/*     15 */	ldd,s	[%xg8+432],%f160


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f56,%f62,%f152,%f62
/*     15 */	fmaddd,s	%f130,%f208,%f76,%f208


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f136,%f68,%f70,%f68
/*     15 */	ldd,s	[%xg8+512],%f190


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f44,%f100,%f104,%f100
/*     15 */	add	%xg8,2080,%xg8


/*     15 */	sxar2
/*     15 */	fmuld,s	%f180,%f48,%f50
/*     15 */	fmaddd,s	%f56,%f232,%f78,%f232


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+112],%f64
/*     15 */	ldd,s	[%xg7+368],%f170


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f46,%f94,%f102,%f94
/*     15 */	fmaddd,s	%f130,%f60,%f244,%f130


/*     15 */	sxar2
/*     15 */	std,s	%f212,[%xg5+832]
/*     15 */	fmaddd,s	%f44,%f96,%f176,%f96


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f46,%f42,%f174,%f42
/*     15 */	std,s	%f66,[%xg5+896]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f54,%f154,%f142,%f156
/*     15 */	fmaddd,s	%f172,%f34,%f160,%f34


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+272],%f72
/*     15 */	fmaddd,s	%f172,%f62,%f158,%f62


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f56,%f208,%f190,%f208
/*     15 */	ldd,s	[%xg4+64],%f116


/*     15 */	sxar2
/*     15 */	fmuld,s	%f182,%f48,%f158
/*     15 */	ldd,s	[%xg8+-1632],%f194


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f56,%f214,%f76,%f214
/*     15 */	ldd,s	[%xg7+320],%f166


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f128,%f118,%f192,%f118
/*     15 */	ldd,s	[%xg4+160],%f90


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f172,%f232,%f152,%f232
/*     15 */	fmaddd,s	%f56,%f130,%f40,%f56


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f172,%f230,%f78,%f230
/*     15 */	ldd,s	[%xg7+224],%f110


/*     15 */	sxar2
/*     15 */	fmuld,s	%f184,%f52,%f74
/*     15 */	fsubd,s	%f84,%f64,%f64


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f128,%f68,%f168,%f68
/*     15 */	fmaddd,s	%f136,%f156,%f162,%f156


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg6+176],%f86
/*     15 */	std,s	%f34,[%xg5+848]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f136,%f154,%f142,%f164
/*     15 */	fmaddd,s	%f172,%f208,%f194,%f208


/*     15 */	sxar2
/*     15 */	std,s	%f62,[%xg5+912]
/*     15 */	fmaddd,s	%f54,%f150,%f148,%f54


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f172,%f214,%f190,%f214
/*     15 */	ldd,s	[%xg4+192],%f40


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f132,%f118,%f176,%f118
/*     15 */	ldd,s	[%xg7+400],%f78


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f44,%f94,%f90,%f94
/*     15 */	std,s	%f232,[%xg5+960]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f138,%f170,%f166,%f168
/*     15 */	fmaddd,s	%f172,%f56,%f252,%f172


/*     15 */	sxar2
/*     15 */	std,s	%f230,[%xg5+1008]
/*     15 */	fmuld,s	%f186,%f52,%f112


/*     15 */	sxar2
/*     15 */	fmuld,s	%f188,%f64,%f32
/*     15 */	fmaddd,s	%f50,%f36,%f86,%f36


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f158,%f146,%f144,%f190
/*     15 */	ldd,s	[%xg8+-1616],%f222


/*     15 */	sxar2
/*     15 */	std,s	%f208,[%xg5+864]
/*     15 */	fmaddd,s	%f132,%f68,%f192,%f68


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f128,%f156,%f174,%f156
/*     15 */	std,s	%f214,[%xg5+928]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f128,%f164,%f162,%f164
/*     15 */	fmaddd,s	%f136,%f54,%f88,%f136


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f122,%f118,%f104,%f118
/*     15 */	fmaddd,s	%f140,%f82,%f80,%f192


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+352],%f76
/*     15 */	std,s	%f172,[%xg5+1024]

/*     15 */	sxar1
/*     15 */	fmaddd,s	%f74,%f168,%f72,%f168

/*     15 */	add	%o0,5,%o0


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f128,%f154,%f142,%f154
/*     15 */	std,s	%f222,[%xg5+880]


/*     15 */	sxar2
/*     15 */	add	%xg5,1040,%xg5
/*     15 */	fmaddd,s	%f134,%f82,%f80,%f194


/*     15 */	sxar2
/*     15 */	sub	%xg0,5,%xg0
/*     15 */	cmp	%xg0,11

/*     15 */	bge,pt	%icc, .L6903
	nop


.L7080:


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg6+384],%f208
/*    ??? */	ldd,s	[%fp+1311],%f142


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f158,%f82,%f80,%f200
/*     15 */	fmaddd,s	%f140,%f190,%f108,%f190


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg6+336],%f206
/*     15 */	ldd,s	[%xg7+384],%f218


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f140,%f146,%f144,%f204
/*     15 */	fmaddd,s	%f138,%f78,%f76,%f242


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+304],%f238
/*     15 */	ldd,s	[%xg7+336],%f214


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f128,%f136,%f120,%f128
/*     15 */	fmaddd,s	%f132,%f154,%f162,%f154


/*     15 */	sxar2
/*    ??? */	ldd,s	[%fp+1327],%f130
/*     15 */	ldd,s	[%xg6+288],%f232


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f134,%f192,%f114,%f192
/*     15 */	fmaddd,s	%f124,%f146,%f144,%f146


/*     15 */	sxar2
/*     15 */	add	%xg8,416,%xg9
/*     15 */	fmuld,s	%f142,%f52,%f198


/*     15 */	sxar2
/*     15 */	fmuld,s	%f142,%f64,%f202
/*    ??? */	ldd,s	[%fp+1359],%f152


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f158,%f208,%f206,%f158
/*     15 */	fmaddd,s	%f140,%f208,%f206,%f210

/*     15 */	add	%o0,4,%o0


/*     15 */	sxar2
/*     15 */	sub	%xg0,4,%xg0
/*     15 */	fmaddd,s	%f140,%f200,%f114,%f200


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f138,%f218,%f214,%f246
/*     15 */	fmuld,s	%f130,%f52,%f196


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+368],%f230
/*     15 */	fmuld,s	%f130,%f64,%f34


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+320],%f226
/*     15 */	ldd,s	[%xg8+272],%f254


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f134,%f190,%f92,%f190
/*     15 */	fmaddd,s	%f134,%f204,%f108,%f204


/*     15 */	sxar2
/*     15 */	fmuld,s	%f152,%f64,%f58
/*     15 */	ldd,s	[%xg8+384],%f142


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f134,%f208,%f206,%f240
/*     15 */	fmaddd,s	%f198,%f170,%f166,%f212


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f198,%f218,%f214,%f216
/*     15 */	ldd,s	[%xg8+336],%f130


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+288],%f236
/*     15 */	fmaddd,s	%f198,%f78,%f76,%f198


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f202,%f230,%f226,%f228
/*     15 */	ldd,s	[%xg7+240],%f248


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+256],%f252
/*     15 */	ldd,s	[%xg6+240],%f244


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f140,%f158,%f232,%f140
/*     15 */	fmaddd,s	%f134,%f210,%f232,%f210


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f196,%f78,%f76,%f220
/*     15 */	fmaddd,s	%f196,%f170,%f166,%f222


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+176],%f172
/*     15 */	fmaddd,s	%f196,%f218,%f214,%f224


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f134,%f200,%f98,%f200
/*     15 */	ldd,s	[%xg7+208],%f180


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg6+208],%f152
/*     15 */	ldd,s	[%xg7+192],%f182


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f132,%f156,%f40,%f156
/*     15 */	fmaddd,s	%f132,%f164,%f174,%f164


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f196,%f212,%f72,%f212
/*     15 */	fmaddd,s	%f196,%f216,%f236,%f216


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg6+192],%f160
/*     15 */	fmaddd,s	%f196,%f198,%f238,%f196


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+144],%f186
/*     15 */	fmaddd,s	%f34,%f228,%f254,%f228


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+224],%f158
/*     15 */	fmaddd,s	%f134,%f140,%f244,%f134


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f132,%f128,%f102,%f128
/*     15 */	ldd,s	[%xg8+400],%f60


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f138,%f220,%f238,%f220
/*     15 */	fmaddd,s	%f138,%f222,%f72,%f222


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+352],%f56
/*     15 */	fmaddd,s	%f138,%f224,%f236,%f224


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f74,%f170,%f166,%f170
/*     15 */	ldd,s	[%xg8+288],%f184


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f74,%f242,%f238,%f242
/*     15 */	fmaddd,s	%f74,%f218,%f214,%f218


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+304],%f178
/*     15 */	fmaddd,s	%f138,%f212,%f110,%f212


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f138,%f216,%f248,%f216
/*     15 */	fmaddd,s	%f138,%f196,%f252,%f138


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+96],%f214
/*     15 */	fmaddd,s	%f74,%f246,%f236,%f246


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f202,%f142,%f130,%f136
/*     15 */	fmaddd,s	%f124,%f208,%f206,%f208


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+256],%f206
/*    ??? */	ldd,s	[%fp+1375],%f162


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+176],%f144
/*     15 */	fmaddd,s	%f74,%f220,%f252,%f220


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f74,%f222,%f110,%f222
/*     15 */	fmaddd,s	%f124,%f194,%f114,%f194


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f74,%f224,%f248,%f224
/*     15 */	ldd,s	[%xg8+128],%f2


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f124,%f192,%f98,%f192
/*     15 */	fmaddd,s	%f132,%f150,%f148,%f132


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+240],%f150
/*     15 */	ldd,s	[%xg6+128],%f38


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f74,%f212,%f172,%f212
/*     15 */	fmaddd,s	%f74,%f216,%f182,%f216


/*     15 */	sxar2
/*     15 */	fmuld,s	%f162,%f64,%f62
/*     15 */	ldd,s	[%xg4+80],%f162


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f74,%f138,%f180,%f138
/*     15 */	ldd,s	[%xg6+160],%f188


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f124,%f190,%f86,%f190
/*     15 */	fmaddd,s	%f124,%f204,%f92,%f204


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg6+144],%f196
/*     15 */	fmaddd,s	%f124,%f134,%f160,%f134


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f124,%f210,%f244,%f210
/*     15 */	fmaddd,s	%f124,%f240,%f232,%f240


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f124,%f200,%f152,%f200
/*    ??? */	ldd,s	[%fp+1391],%f66


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f74,%f78,%f76,%f74
/*     15 */	ldd,s	[%xg8+208],%f78


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f106,%f146,%f108,%f146
/*     15 */	fmaddd,s	%f32,%f228,%f158,%f228


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+192],%f108
/*     15 */	fmaddd,s	%f112,%f218,%f236,%f218


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+128],%f198
/*     15 */	fmaddd,s	%f122,%f154,%f174,%f154


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f106,%f208,%f232,%f208
/*     15 */	fmaddd,s	%f34,%f230,%f226,%f70


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+160],%f140
/*     15 */	fmaddd,s	%f124,%f82,%f80,%f124


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+144],%f148
/*     15 */	fmaddd,s	%f34,%f136,%f184,%f136


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f34,%f142,%f130,%f166
/*     15 */	ldd,s	[%xg6+64],%f0


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f202,%f60,%f56,%f202
/*     15 */	fmaddd,s	%f34,%f60,%f56,%f236


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f32,%f230,%f226,%f234
/*     15 */	ldd,s	[%xg6+80],%f80


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f122,%f68,%f176,%f68
/*     15 */	ldd,s	[%xg4],%f4


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f122,%f156,%f186,%f156
/*     15 */	fmaddd,s	%f122,%f164,%f40,%f164


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+16],%f174
/*     15 */	fmaddd,s	%f122,%f126,%f102,%f126


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+32],%f232
/*     15 */	fmaddd,s	%f122,%f128,%f90,%f128


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f106,%f194,%f98,%f194
/*     15 */	fmuld,s	%f66,%f52,%f54


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg4+48],%f14
/*     15 */	mov	%xg9,%xg4


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f106,%f190,%f38,%f190
/*     15 */	fmaddd,s	%f106,%f204,%f86,%f204


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f106,%f134,%f196,%f134
/*     15 */	fmaddd,s	%f34,%f202,%f178,%f34


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f106,%f210,%f160,%f210
/*     15 */	fmaddd,s	%f106,%f192,%f152,%f192


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+64],%f10
/*     15 */	fmaddd,s	%f106,%f240,%f244,%f240


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f106,%f200,%f188,%f200
/*     15 */	ldd,s	[%xg8+96],%f20


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f58,%f230,%f226,%f230
/*     15 */	ldd,s	[%xg8],%f18


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f58,%f228,%f144,%f228
/*     15 */	fmaddd,s	%f122,%f132,%f88,%f122


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f112,%f170,%f72,%f170
/*     15 */	fmaddd,s	%f32,%f60,%f56,%f250


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f32,%f70,%f254,%f70
/*     15 */	fmaddd,s	%f32,%f136,%f150,%f136


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f32,%f166,%f184,%f166
/*     15 */	fmaddd,s	%f32,%f142,%f130,%f202


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f32,%f236,%f178,%f236
/*     15 */	fmaddd,s	%f112,%f168,%f110,%f168


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f32,%f34,%f206,%f34
/*     15 */	ldd,s	[%xg8+32],%f32


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg6+96],%f82
/*     15 */	fmaddd,s	%f112,%f242,%f252,%f242


/*     15 */	sxar2
/*     15 */	fmuld,s	%f66,%f64,%f66
/*     15 */	ldd,s	[%xg7+64],%f226


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f106,%f124,%f114,%f106
/*     15 */	fmaddd,s	%f112,%f212,%f198,%f212


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+80],%f76
/*     15 */	fmaddd,s	%f112,%f222,%f172,%f222


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f112,%f216,%f148,%f216
/*     15 */	ldd,s	[%xg7+96],%f176


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f112,%f224,%f182,%f224
/*     15 */	fmaddd,s	%f112,%f220,%f180,%f220


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg6],%f6
/*     15 */	fmaddd,s	%f112,%f246,%f248,%f246


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f112,%f138,%f140,%f138
/*     15 */	fmaddd,s	%f58,%f234,%f254,%f234


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg6+16],%f88
/*     15 */	fmaddd,s	%f62,%f228,%f2,%f228


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f58,%f250,%f178,%f250
/*     15 */	ldd,s	[%xg6+32],%f132


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7],%f72
/*     15 */	fmaddd,s	%f58,%f70,%f158,%f70


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg7+16],%f8
/*     15 */	ldd,s	[%xg7+32],%f12


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f58,%f142,%f130,%f142
/*     15 */	fmaddd,s	%f58,%f136,%f108,%f136


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+160],%f114
/*     15 */	fmaddd,s	%f58,%f166,%f150,%f166


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f58,%f202,%f184,%f202
/*     15 */	ldd,s	[%xg8+144],%f124


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f58,%f236,%f206,%f236
/*     15 */	fmaddd,s	%f58,%f34,%f78,%f34


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f62,%f230,%f254,%f230
/*     15 */	fmaddd,s	%f112,%f74,%f238,%f112


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+80],%f74
/*     15 */	fmaddd,s	%f62,%f234,%f158,%f234


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+16],%f238
/*     15 */	fmaddd,s	%f66,%f228,%f10,%f228


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f62,%f250,%f206,%f250
/*     15 */	fmaddd,s	%f62,%f70,%f144,%f70


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f62,%f142,%f184,%f142
/*     15 */	fmaddd,s	%f62,%f136,%f124,%f136


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f62,%f166,%f108,%f166
/*     15 */	fmaddd,s	%f62,%f202,%f150,%f202


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f62,%f236,%f78,%f236
/*     15 */	fmaddd,s	%f58,%f60,%f56,%f58


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg6+48],%f254
/*     15 */	fmaddd,s	%f46,%f164,%f186,%f164


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f62,%f34,%f114,%f34
/*     15 */	ldd,s	[%xg7+48],%f16


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f46,%f118,%f116,%f118
/*     15 */	fmaddd,s	%f46,%f68,%f104,%f68


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f50,%f146,%f92,%f146
/*     15 */	fmaddd,s	%f46,%f156,%f162,%f156


/*     15 */	sxar2
/*     15 */	std,s	%f100,[%xg5+112]
/*     15 */	fmaddd,s	%f46,%f154,%f40,%f154


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f50,%f194,%f152,%f194
/*     15 */	std,s	%f96,[%xg5+160]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f50,%f208,%f244,%f208
/*     15 */	fmaddd,s	%f46,%f128,%f214,%f128


/*     15 */	sxar2
/*     15 */	std,s	%f94,[%xg5+144]
/*     15 */	fmaddd,s	%f46,%f126,%f90,%f126


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f54,%f168,%f172,%f168
/*     15 */	fmaddd,s	%f50,%f190,%f0,%f190


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f50,%f204,%f38,%f204
/*     15 */	fmaddd,s	%f54,%f170,%f110,%f170


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f50,%f134,%f80,%f134
/*     15 */	fmaddd,s	%f50,%f210,%f196,%f210


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f50,%f240,%f160,%f240
/*     15 */	fmaddd,s	%f54,%f242,%f180,%f242


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f54,%f218,%f248,%f218
/*     15 */	fmaddd,s	%f50,%f200,%f82,%f200


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f50,%f192,%f188,%f192
/*     15 */	fmaddd,s	%f66,%f234,%f144,%f234


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f54,%f212,%f226,%f212
/*     15 */	fmaddd,s	%f54,%f222,%f198,%f222


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f66,%f230,%f158,%f230
/*     15 */	fmaddd,s	%f54,%f216,%f76,%f216


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f54,%f224,%f148,%f224
/*     15 */	fmaddd,s	%f54,%f246,%f182,%f246


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f66,%f250,%f78,%f250
/*     15 */	fmaddd,s	%f54,%f138,%f176,%f138


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f54,%f220,%f140,%f220
/*     15 */	fmaddd,s	%f66,%f70,%f2,%f70


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f66,%f142,%f150,%f142
/*     15 */	fmaddd,s	%f66,%f136,%f74,%f136


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f66,%f166,%f124,%f166
/*     15 */	fmaddd,s	%f66,%f202,%f108,%f202


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f62,%f58,%f178,%f58
/*     15 */	fmaddd,s	%f66,%f34,%f20,%f34


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f66,%f236,%f114,%f236
/*     15 */	fmaddd,s	%f44,%f164,%f162,%f164


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f64,%f228,%f18,%f228
/*     15 */	fmaddd,s	%f46,%f122,%f120,%f46


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f50,%f106,%f98,%f50
/*     15 */	fmaddd,s	%f54,%f112,%f252,%f54


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f44,%f118,%f4,%f118
/*     15 */	fmaddd,s	%f44,%f68,%f116,%f68


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f48,%f36,%f38,%f38
/*     15 */	fmaddd,s	%f48,%f146,%f86,%f146


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f66,%f58,%f206,%f66
/*     15 */	fmaddd,s	%f44,%f156,%f174,%f156


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f44,%f154,%f186,%f154
/*     15 */	fmaddd,s	%f44,%f42,%f40,%f42


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f44,%f128,%f232,%f128
/*     15 */	std,s	%f164,[%xg5+80]


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f44,%f126,%f214,%f126
/*     15 */	fmaddd,s	%f48,%f194,%f188,%f194


/*     15 */	sxar2
/*     15 */	ldd,s	[%xg8+48],%f164
/*     15 */	fmaddd,s	%f44,%f46,%f102,%f46


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f48,%f190,%f6,%f190
/*     15 */	fmaddd,s	%f48,%f204,%f0,%f204


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f52,%f168,%f198,%f168
/*     15 */	fmaddd,s	%f52,%f170,%f172,%f170


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f48,%f134,%f88,%f134
/*     15 */	fmaddd,s	%f48,%f210,%f80,%f210


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f48,%f240,%f196,%f240
/*     15 */	fmaddd,s	%f48,%f208,%f160,%f208


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f48,%f200,%f132,%f200
/*     15 */	fmaddd,s	%f48,%f192,%f82,%f192


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f52,%f242,%f140,%f242
/*     15 */	fmaddd,s	%f48,%f50,%f152,%f50


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f52,%f212,%f72,%f212
/*     15 */	fmaddd,s	%f52,%f222,%f226,%f222


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f64,%f234,%f2,%f234
/*     15 */	fmaddd,s	%f64,%f230,%f144,%f230


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f52,%f216,%f8,%f216
/*     15 */	fmaddd,s	%f52,%f224,%f76,%f224


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f52,%f246,%f148,%f246
/*     15 */	fmaddd,s	%f52,%f218,%f182,%f218


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f52,%f138,%f12,%f138
/*     15 */	fmaddd,s	%f52,%f220,%f176,%f220


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f64,%f250,%f114,%f250
/*     15 */	fmaddd,s	%f52,%f54,%f180,%f54


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f64,%f70,%f10,%f70
/*     15 */	fmaddd,s	%f64,%f136,%f238,%f136


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f64,%f166,%f74,%f166
/*     15 */	fmaddd,s	%f64,%f202,%f124,%f202


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f64,%f142,%f108,%f142
/*     15 */	fmaddd,s	%f64,%f34,%f32,%f34


/*     15 */	sxar2
/*     15 */	fmaddd,s	%f64,%f236,%f20,%f236
/*     15 */	fmaddd,s	%f64,%f66,%f78,%f66


/*     15 */	sxar2
/*     15 */	std,s	%f118,[%xg5]
/*     15 */	std,s	%f68,[%xg5+64]


/*     15 */	sxar2
/*     15 */	std,s	%f156,[%xg5+16]
/*     15 */	std,s	%f154,[%xg5+128]


/*     15 */	sxar2
/*     15 */	std,s	%f42,[%xg5+176]
/*     15 */	std,s	%f128,[%xg5+32]


/*     15 */	sxar2
/*     15 */	std,s	%f126,[%xg5+96]
/*     15 */	std,s	%f46,[%xg5+192]


/*     15 */	sxar2
/*     15 */	std,s	%f14,[%xg5+48]
/*     15 */	std,s	%f38,[%xg5+320]


/*     15 */	sxar2
/*     15 */	std,s	%f146,[%xg5+368]
/*     15 */	std,s	%f194,[%xg5+352]


/*     15 */	sxar2
/*     15 */	std,s	%f190,[%xg5+208]
/*     15 */	std,s	%f204,[%xg5+272]


/*     15 */	sxar2
/*     15 */	std,s	%f134,[%xg5+224]
/*     15 */	std,s	%f210,[%xg5+288]


/*     15 */	sxar2
/*     15 */	std,s	%f240,[%xg5+336]
/*     15 */	std,s	%f208,[%xg5+384]


/*     15 */	sxar2
/*     15 */	std,s	%f200,[%xg5+240]
/*     15 */	std,s	%f192,[%xg5+304]


/*     15 */	sxar2
/*     15 */	std,s	%f50,[%xg5+400]
/*     15 */	std,s	%f254,[%xg5+256]


/*     15 */	sxar2
/*     15 */	std,s	%f168,[%xg5+528]
/*     15 */	std,s	%f170,[%xg5+576]


/*     15 */	sxar2
/*     15 */	std,s	%f242,[%xg5+560]
/*     15 */	std,s	%f212,[%xg5+416]


/*     15 */	sxar2
/*     15 */	std,s	%f222,[%xg5+480]
/*     15 */	std,s	%f216,[%xg5+432]


/*     15 */	sxar2
/*     15 */	std,s	%f224,[%xg5+496]
/*     15 */	std,s	%f246,[%xg5+544]


/*     15 */	sxar2
/*     15 */	std,s	%f218,[%xg5+592]
/*     15 */	std,s	%f138,[%xg5+448]


/*     15 */	sxar2
/*     15 */	std,s	%f220,[%xg5+512]
/*     15 */	std,s	%f54,[%xg5+608]


/*     15 */	sxar2
/*     15 */	std,s	%f16,[%xg5+464]
/*     15 */	std,s	%f234,[%xg5+736]


/*     15 */	sxar2
/*     15 */	std,s	%f230,[%xg5+784]
/*     15 */	std,s	%f250,[%xg5+768]


/*     15 */	sxar2
/*     15 */	std,s	%f228,[%xg5+624]
/*     15 */	std,s	%f70,[%xg5+688]


/*     15 */	sxar2
/*     15 */	std,s	%f136,[%xg5+640]
/*     15 */	std,s	%f166,[%xg5+704]


/*     15 */	sxar2
/*     15 */	std,s	%f202,[%xg5+752]
/*     15 */	std,s	%f142,[%xg5+800]


/*     15 */	sxar2
/*     15 */	std,s	%f34,[%xg5+656]
/*     15 */	std,s	%f236,[%xg5+720]


/*     15 */	sxar2
/*     15 */	std,s	%f164,[%xg5+672]
/*     15 */	std,s	%f66,[%xg5+816]

/*     15 */	sxar1
/*     15 */	add	%xg5,832,%xg5

.L7075:


.L7074:


.L7077:


/*     38 */	sxar2
/*     38 */	ldd,s	[%xg4+112],%f196
/* #00001 */	ldd,s	[%fp+1391],%f60



/*     38 */	sxar2
/*     38 */	subcc	%xg0,1,%xg0
/* #00001 */	ldd,s	[%fp+1375],%f62


/*     38 */	sxar2
/* #00001 */	ldd,s	[%fp+1359],%f64
/* #00001 */	ldd,s	[%fp+1343],%f66


/*     19 */	sxar2
/* #00001 */	ldd,s	[%fp+1327],%f68
/*     19 */	ldd,s	[%xg4+368],%f214


/*     35 */	sxar2
/* #00001 */	ldd,s	[%fp+1311],%f70
/*     35 */	fsubd,s	%f84,%f196,%f196


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg4+320],%f210
/*     21 */	ldd,s	[%xg4+272],%f218


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg4+224],%f220
/*     21 */	ldd,s	[%xg4+176],%f222


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg4+128],%f224
/*     21 */	ldd,s	[%xg4+64],%f226


/*     38 */	sxar2
/*     38 */	ldd,s	[%xg4],%f228
/*     38 */	fmuld,s	%f68,%f196,%f206


/*     38 */	sxar2
/*     38 */	fmuld,s	%f70,%f196,%f208
/*     38 */	fmuld,s	%f64,%f196,%f202


/*     19 */	sxar2
/*     19 */	fmuld,s	%f66,%f196,%f204
/*     19 */	ldd,s	[%xg4+384],%f236


/*     38 */	sxar2
/*     38 */	ldd,s	[%xg4+336],%f232
/*     38 */	fmuld,s	%f60,%f196,%f198


/*     21 */	sxar2
/*     21 */	fmuld,s	%f62,%f196,%f200
/*     21 */	ldd,s	[%xg4+288],%f240


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg4+400],%f32
/*     21 */	ldd,s	[%xg4+352],%f254


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg4+304],%f36
/*     21 */	ldd,s	[%xg4+240],%f242


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg4+256],%f38
/*     21 */	ldd,s	[%xg4+192],%f244


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg4+208],%f40
/*     21 */	fmaddd,s	%f208,%f214,%f210,%f212


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg4+144],%f246
/*     21 */	fmaddd,s	%f208,%f236,%f232,%f234


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f206,%f214,%f210,%f216
/*     21 */	fmaddd,s	%f206,%f236,%f232,%f238


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg4+160],%f42
/*     21 */	ldd,s	[%xg4+80],%f248


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f208,%f32,%f254,%f208
/*     21 */	fmaddd,s	%f206,%f32,%f254,%f34


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f204,%f214,%f210,%f230
/*     21 */	fmaddd,s	%f204,%f236,%f232,%f252


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg4+96],%f44
/*     21 */	fmaddd,s	%f202,%f214,%f210,%f214


/*     21 */	sxar2
/*     21 */	ldd,s	[%xg4+16],%f250
/*     21 */	fmaddd,s	%f202,%f236,%f232,%f236


/*     19 */	sxar2
/*     19 */	ldd,s	[%xg4+32],%f46
/*     19 */	ldd,s	[%xg4+48],%f72


/*     21 */	sxar2
/*     21 */	add	%xg4,416,%xg4
/*     21 */	fmaddd,s	%f206,%f212,%f218,%f212


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f206,%f234,%f240,%f234
/*     21 */	fmaddd,s	%f204,%f216,%f218,%f216


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f204,%f238,%f240,%f238
/*     21 */	fmaddd,s	%f206,%f208,%f36,%f206


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f204,%f34,%f36,%f34
/*     21 */	fmaddd,s	%f202,%f230,%f218,%f230


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f202,%f252,%f240,%f252
/*     21 */	fmaddd,s	%f200,%f214,%f218,%f214


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f200,%f236,%f240,%f236
/*     21 */	std,s	%f72,[%xg5+48]


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f204,%f212,%f220,%f212
/*     21 */	fmaddd,s	%f204,%f234,%f242,%f234


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f202,%f216,%f220,%f216
/*     21 */	fmaddd,s	%f202,%f238,%f242,%f238


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f204,%f206,%f38,%f206
/*     21 */	fmaddd,s	%f204,%f32,%f254,%f204


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f202,%f34,%f38,%f34
/*     21 */	fmaddd,s	%f202,%f32,%f254,%f32


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f200,%f230,%f220,%f230
/*     21 */	fmaddd,s	%f200,%f252,%f242,%f252


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f198,%f214,%f220,%f214
/*     21 */	fmaddd,s	%f198,%f236,%f242,%f236


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f202,%f212,%f222,%f212
/*     21 */	fmaddd,s	%f202,%f234,%f244,%f234


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f200,%f216,%f222,%f216
/*     21 */	fmaddd,s	%f200,%f238,%f244,%f238


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f202,%f206,%f40,%f206
/*     21 */	fmaddd,s	%f202,%f204,%f36,%f202


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f200,%f34,%f40,%f34
/*     21 */	fmaddd,s	%f200,%f32,%f36,%f32


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f198,%f230,%f222,%f230
/*     21 */	fmaddd,s	%f198,%f252,%f244,%f252


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f196,%f214,%f222,%f214
/*     21 */	fmaddd,s	%f196,%f236,%f244,%f236


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f200,%f212,%f224,%f212
/*     21 */	fmaddd,s	%f200,%f234,%f246,%f234


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f198,%f216,%f224,%f216
/*     21 */	fmaddd,s	%f198,%f238,%f246,%f238


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f200,%f206,%f42,%f206
/*     21 */	fmaddd,s	%f200,%f202,%f38,%f200


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f198,%f34,%f42,%f34
/*     21 */	fmaddd,s	%f198,%f32,%f38,%f32


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f196,%f230,%f224,%f230
/*     21 */	fmaddd,s	%f196,%f252,%f246,%f252


/*     21 */	sxar2
/*     21 */	std,s	%f214,[%xg5+160]
/*     21 */	fmaddd,s	%f198,%f212,%f226,%f212


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f198,%f234,%f248,%f234
/*     21 */	fmaddd,s	%f196,%f216,%f226,%f216


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f196,%f238,%f248,%f238
/*     21 */	fmaddd,s	%f198,%f206,%f44,%f206


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f198,%f200,%f40,%f198
/*     21 */	fmaddd,s	%f196,%f34,%f44,%f34


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f196,%f32,%f40,%f32
/*     21 */	std,s	%f230,[%xg5+112]


/*     21 */	sxar2
/*     21 */	std,s	%f236,[%xg5+176]
/*     21 */	fmaddd,s	%f196,%f212,%f228,%f212


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f196,%f234,%f250,%f234
/*     21 */	std,s	%f216,[%xg5+64]


/*     21 */	sxar2
/*     21 */	fmaddd,s	%f196,%f206,%f46,%f206
/*     21 */	fmaddd,s	%f196,%f198,%f42,%f196


/*     21 */	sxar2
/*     21 */	std,s	%f238,[%xg5+80]
/*     21 */	std,s	%f252,[%xg5+128]


/*     21 */	sxar2
/*     21 */	std,s	%f34,[%xg5+96]
/*     21 */	std,s	%f212,[%xg5]


/*     21 */	sxar2
/*     21 */	std,s	%f234,[%xg5+16]
/*     21 */	std,s	%f206,[%xg5+32]


/*     21 */	sxar2
/*     21 */	std,s	%f32,[%xg5+192]
/*     21 */	std,s	%f196,[%xg5+144]

/*     59 */	sxar1
/*     59 */	add	%xg5,208,%xg5

/*     59 */	bne,pt	%icc, .L7077
/*     59 */	add	%o0,1,%o0


.L7073:

/*     59 */
/*     59 */	ba	.L6901
	nop


.L6913:

/*     59 *//*     59 */	call	__mpc_obar
/*     59 */	ldx	[%fp+2199],%o0

/*     59 *//*     59 */	call	__mpc_obar
/*     59 */	ldx	[%fp+2199],%o0


.L6914:

/*     59 */	ret
	restore



.LLFE5:
	.size	_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE._OMP_1,.-_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE._OMP_1
	.type	_ZN7Gravity14predict_all_rpEidPKNS_9GParticleEPNS_10GPredictorE._OMP_1,#function
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite8-k.cpp _ZN7Gravity19calc_force_in_rangeEiidP5Force $"
	.section	".text"
	.global	_ZN7Gravity19calc_force_in_rangeEiidP5Force
	.align	64
_ZN7Gravity19calc_force_in_rangeEiidP5Force:
.LLFB6:
.L569:

/*     62 */	save	%sp,-2960,%sp
.LLCFI4:
/*     62 */	stw	%i2,[%fp+2195]
/*     62 */	stx	%i0,[%fp+2175]
/*     62 */	stw	%i1,[%fp+2187]
/*     62 */	std	%f6,[%fp+2199]
/*     62 */	stx	%i4,[%fp+2207]

.L570:

/*     67 */	sethi	%h44(.LB0..127.1),%l0

/*     67 */	or	%l0,%m44(.LB0..127.1),%l0

/*     67 */	sllx	%l0,12,%l0

/*     67 */	or	%l0,%l44(.LB0..127.1),%l0


/*     67 */	sxar2
/*     67 */	ldsb	[%l0],%xg0
/*     67 */	cmp	%xg0,%g0

/*     67 */	bne,pt	%icc, .L572
	nop


.L571:


.LLEHB0:
/*     67 */	call	__cxa_guard_acquire
/*     67 */	mov	%l0,%o0
.LLEHE0:


.L6720:

/*     67 */	cmp	%o0,%g0

/*     67 */	be	.L572
	nop


.L573:

/*     68 */	sethi	%h44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%o0

/*     68 */	sethi	%h44(_ZN7Gravity6GForceC1Ev),%o3

/*     68 */	or	%o0,%m44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%o0

/*     68 */	or	%o3,%m44(_ZN7Gravity6GForceC1Ev),%o3

/*     68 */	sllx	%o0,12,%o0

/*     68 */	sllx	%o3,12,%o3

/*     68 */	or	%o0,%l44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%o0

/*     68 */	or	%o3,%l44(_ZN7Gravity6GForceC1Ev),%o3

/*     68 */	sethi	%hi(8192),%o1

/*     68 */	mov	192,%o2


.LLEHB1:
/*     68 */	call	__cxa_vec_ctor
/*     68 */	mov	%g0,%o4
.LLEHE1:


.L6873:

/*     68 */	ba	.L6719
	nop


.L576:

/*     68 */

.L577:


/*     68 */	call	__cxa_guard_abort
/*     68 */	mov	%l0,%o0


.L6718:


.LLEHB2:
/*     68 */	call	_Unwind_Resume
/*     68 */	mov	%i0,%o0


.L6719:


/*     68 */	call	__cxa_guard_release
/*     68 */	mov	%l0,%o0
.LLEHE2:


.L572:

/*     70 *//*     70 */	sethi	%h44(_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_2),%o0
/*     70 */	mov	%fp,%l1
/*     70 */	or	%o0,%m44(_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_2),%o0
/*     70 */	mov	%g0,%l2
/*     70 */	sllx	%o0,12,%o0
/*     70 */	mov	%l1,%o1
/*     70 */	or	%o0,%l44(_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_2),%o0
/*     70 */	call	__mpc_opar
/*     70 */	mov	%l2,%o2

/*    185 */
/*    187 *//*    187 */	sethi	%h44(_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_3),%o0
/*    187 */	mov	%l1,%o1
/*    187 */	or	%o0,%m44(_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_3),%o0
/*    187 */	mov	%l2,%o2
/*    187 */	sllx	%o0,12,%o0
/*    187 */	call	__mpc_opar
/*    187 */	or	%o0,%l44(_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_3),%o0

/*    234 */
/*    234 */	ret
	restore



.L632:


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
	.uleb128	.L576-.LLFB6
	.uleb128	0x0
	.uleb128	.LLEHB2-.LLFB6
	.uleb128	.LLEHE2-.LLEHB2
	.uleb128	0x0
	.uleb128	0x0
.LLLSDACSE6:
	.sleb128	0
	.sleb128	0
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite8-k.cpp _ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_2 $"
	.section	".text"
	.align	64
_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_2:
.LLFB7:
.L6915:

/*     70 */	sethi	%hi(10240),%g1
	xor	%g1,-320,%g1
	save	%sp,%g1,%sp
.LLCFI5:
/*     70 */	stx	%i0,[%fp+2175]
/*     70 */	stx	%i1,[%fp+2183]
/*     70 */	stx	%i2,[%fp+2191]
/*     70 */	stx	%i3,[%fp+2199]
/*     70 */	stx	%i0,[%fp+2175]

.L6916:

/*     70 *//*     70 */	sxar1
/*     70 */	ldsw	[%i0+2031],%xg3
/*     70 */
.LLEHB3:
/*     72 */	call	omp_get_thread_num
	nop
/*     72 */	mov	%o0,%l2

.L6917:

/*     73 */
/*     73 */	call	__mpc_pmnm
	nop
/*     73 */	sxar2
/*     73 */	ldx	[%fp+2191],%xg2
/*     73 */	cmp	%xg2,%o0
/*     73 */	bne,pt	%xcc, .L6923
	nop


.L6918:

/*     73 */
/*     74 */	call	omp_get_num_threads
	nop
.LLEHE3:


.L6919:

/*     74 */	ba	.L6922
	nop


.L6920:


.L6921:

/*      0 */	call	_ZSt9terminatev
	nop


.L6922:

/*     74 */	stw	%o0,[%i0+2027]

.L6923:

/*     74 */
/*     76 */	sxar1
/*     76 */	ldx	[%i0+2175],%xg0

/*     77 */	ldsw	[%i0+2187],%l0

/*     76 */	sxar2
/*     76 */	ldsw	[%i0+2195],%xg1
/*     76 */	ldsw	[%xg0],%l1

/*     77 */	sxar1
/*     77 */	cmp	%l0,%xg1
/*     77 */	bge	.L6941
	nop


.L6924:

/*     77 */	sxar1
/*     77 */	fzero,s	%f188

/*    ??? */	sethi	%hi(8208),%o7

/*    ??? */	xor	%o7,-17,%o7

/*     77 */	sxar1
/*    ??? */	std,s	%f188,[%fp+%o7]

.L6925:

/*    104 */	sethi	%h44(.LR0.cnt.6),%g1

/*    104 */	sethi	%h44(.LR0.cnt.7),%g2

/*    104 */	or	%g1,%m44(.LR0.cnt.6),%g1

/*    104 */	or	%g2,%m44(.LR0.cnt.7),%g2

/*    104 */	sllx	%g1,12,%g1

/*    ??? */	sethi	%hi(8160),%o2

/*    104 */	or	%g1,%l44(.LR0.cnt.6),%g1

/*    104 */	sllx	%g2,12,%g2

/*    104 */	or	%g2,%l44(.LR0.cnt.7),%g2


/*    104 */	sxar2
/*    104 */	ldd	[%g1],%f118
/*    104 */	ldd	[%g1],%f374


/*    ??? */	xor	%o2,-993,%o2

/*    104 */	sethi	%h44(.LR0.cnt.9),%g3


/*    104 */	sxar2
/*    104 */	ldd	[%g2],%f120
/*    104 */	ldd	[%g2],%f376


/*    ??? */	sethi	%hi(8176),%o3

/*    104 */	or	%g3,%m44(.LR0.cnt.9),%g3

/*    ??? */	xor	%o3,-1009,%o3

/*    104 */	sllx	%g3,12,%g3

/*    104 */	sxar1
/*    104 */	sethi	%h44(.LR0.cnt.10),%xg0

/*    104 */	or	%g3,%l44(.LR0.cnt.9),%g3


/*    104 */	sxar2
/*    104 */	or	%xg0,%m44(.LR0.cnt.10),%xg0
/*    ??? */	std,s	%f118,[%fp+%o2]


/*    104 */	sxar2
/*    104 */	sllx	%xg0,12,%xg0
/*    ??? */	std,s	%f120,[%fp+%o3]

/*    ??? */	sethi	%hi(8192),%o4


/*    104 */	sxar2
/*    104 */	or	%xg0,%l44(.LR0.cnt.10),%xg0
/*    104 */	ldd	[%g3],%f122

/*    104 */	sxar1
/*    104 */	ldd	[%g3],%f378


/*    ??? */	xor	%o4,-1,%o4

/*    104 */	srl	%l1,31,%l3


/*    104 */	sxar2
/*    104 */	ldd	[%xg0],%f124
/*    104 */	ldd	[%xg0],%f380


/*    ??? */	sethi	%hi(8224),%o5

/*    104 */	sra	%l2,%g0,%l2

/*    ??? */	xor	%o5,-33,%o5

/*    104 */	add	%l3,%l1,%l3

/*    104 */	sethi	%h44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%l7

/*    104 */	add	%l2,%l2,%l4

/*    104 */	sra	%l3,1,%l3

/*    104 */	sxar1
/*    ??? */	std,s	%f122,[%fp+%o4]

/*    104 */	add	%l4,%l2,%l4

/*    104 */	or	%l7,%m44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%l7

/*    104 */	sxar1
/*    ??? */	std,s	%f124,[%fp+%o5]

/*    104 */	add	%l3,%l3,%l5

/*    104 */	mov	1,%i1

/*    104 */	sub	%l1,%l5,%l5

/*    104 */	sllx	%l7,12,%l7

/*    104 */	sllx	%l4,15,%l4

/*    104 */	add	%l3,1,%l6

/*    104 */	or	%l7,%l44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%l7

/*    104 */	add	%fp,-993,%i2

/*    104 */	sra	%i1,%g0,%i3

.L6926:

/*    ??? */	sethi	%hi(8208),%o0


/*     26 */	sxar2
/*     26 */	srl	%l0,31,%xg0
/*     26 */	ldd	[%i0+2199],%f162

/*    ??? */	xor	%o0,-17,%o0


/*     26 */	sxar2
/*     26 */	add	%xg0,%l0,%xg0
/*     26 */	ldd	[%i0+2199],%f418



/*     90 */	sxar2
/*    ??? */	ldd,s	[%fp+%o0],%f160
/*     90 */	sra	%xg0,1,%xg0


/*     90 */	sxar2
/*     90 */	sra	%xg0,%g0,%xg0
/*     90 */	mulx	%xg0,208,%xg0


/*     34 */	sxar2
/*     34 */	std,s	%f160,[%i2+-4032]
/*     34 */	std,s	%f160,[%i2+-4016]


/*     34 */	sxar2
/*     34 */	std,s	%f160,[%i2+-4000]
/*     34 */	std,s	%f160,[%i2+-3984]


/*     34 */	sxar2
/*     34 */	std,s	%f160,[%i2+-3968]
/*     34 */	std,s	%f160,[%i2+-3952]


/*     34 */	sxar2
/*     34 */	std,s	%f160,[%i2+-3936]
/*     34 */	std,s	%f160,[%i2+-3920]


/*     34 */	sxar2
/*     34 */	std,s	%f160,[%i2+-3904]
/*     34 */	std,s	%f160,[%i2+-3888]


/*     34 */	sxar2
/*     34 */	std,s	%f160,[%i2+-3872]
/*     34 */	std,s	%f160,[%i2+-3856]


/*     34 */	sxar2
/*     34 */	std,s	%f160,[%i2+-3840]
/*     34 */	std,s	%f160,[%i2+-3824]


/*     34 */	sxar2
/*     34 */	std,s	%f160,[%i2+-3808]
/*     34 */	std,s	%f160,[%i2+-3792]


/*     34 */	sxar2
/*     34 */	std,s	%f160,[%i2+-3776]
/*     34 */	std,s	%f160,[%i2+-3760]


/*     34 */	sxar2
/*     34 */	std,s	%f160,[%i2+-3744]
/*     34 */	std,s	%f160,[%i2+-3728]


/*     34 */	sxar2
/*     34 */	std,s	%f160,[%i2+-3712]
/*     34 */	std,s	%f160,[%i2+-3696]


/*     34 */	sxar2
/*     34 */	std,s	%f160,[%i2+-3680]
/*     34 */	std,s	%f160,[%i2+-3664]

/*     90 */	ldx	[%i0+2175],%o1


/*     19 */	sxar2
/*     19 */	ldx	[%o1+16],%xg1
/*     19 */	add	%xg1,%xg0,%xg1


/*     19 */	sxar2
/*     19 */	ldd,s	[%xg1],%f164
/*     19 */	std,s	%f164,[%i2+-3648]


/*     19 */	sxar2
/*     19 */	ldd,s	[%xg1+16],%f166
/*     19 */	std,s	%f166,[%i2+-3632]


/*     19 */	sxar2
/*     19 */	ldd,s	[%xg1+32],%f168
/*     19 */	std,s	%f168,[%i2+-3616]


/*     19 */	sxar2
/*     19 */	ldd,s	[%xg1+64],%f170
/*     19 */	std,s	%f170,[%i2+-3600]


/*     19 */	sxar2
/*     19 */	ldd,s	[%xg1+80],%f172
/*     19 */	std,s	%f172,[%i2+-3584]


/*     19 */	sxar2
/*     19 */	ldd,s	[%xg1+96],%f174
/*     19 */	std,s	%f174,[%i2+-3568]


/*     19 */	sxar2
/*     19 */	ldd,s	[%xg1+112],%f176
/*     19 */	std,s	%f176,[%i2+-3552]


/*     19 */	sxar2
/*     19 */	ldd,s	[%xg1+128],%f178
/*     19 */	std,s	%f178,[%i2+-3536]


/*     19 */	sxar2
/*     19 */	ldd,s	[%xg1+144],%f180
/*     19 */	std,s	%f180,[%i2+-3520]


/*     19 */	sxar2
/*     19 */	ldd,s	[%xg1+160],%f182
/*     19 */	std,s	%f182,[%i2+-3504]


/*     19 */	sxar2
/*     19 */	ldd,s	[%xg1+176],%f184
/*     19 */	std,s	%f184,[%i2+-3488]


/*     26 */	sxar2
/*     26 */	ldd,s	[%xg1+192],%f186
/*     26 */	std,s	%f162,[%i2+-3456]

/*     19 */	sxar1
/*     19 */	std,s	%f186,[%i2+-3472]

/*    103 */
/*    103 */
/*    104 */	cmp	%l1,%g0
/*    104 */	ble	.L6939
	nop


.L6927:

/*    104 */	cmp	%l5,%g0

/*    104 */	sxar1
/*    104 */	mov	%l3,%xg3

/*    104 */	be	.L6929
	nop


.L6928:

/*    104 */	sxar1
/*    104 */	mov	%l6,%xg3

.L6929:


/*    104 */	sxar2
/*    104 */	ldx	[%fp+2183],%xg4
/*    104 */	ldx	[%fp+2191],%xg5


/*    104 */	sxar2
/*    104 */	sra	%xg3,%g0,%xg2
/*    104 */	sra	%xg4,%g0,%xg4


/*    104 */	sxar2
/*    104 */	sra	%xg5,%g0,%xg5
/*    104 */	sra	%xg4,%g0,%xg6


/*    104 */	sxar2
/*    104 */	sdivx	%xg2,%xg6,%xg2

/*    104 */	sra	%xg2,%g0,%xg2


/*    104 */	sxar2
/*    104 */	mulx	%xg4,%xg2,%xg4
/*    104 */	subcc	%xg3,%xg4,%xg3

/*    104 */	bne,pt	%icc, .L6931
	nop


.L6930:


/*    104 */	sxar2
/*    104 */	add	%xg5,%xg5,%xg5
/*    104 */	add	%xg2,%xg2,%xg7


/*    104 */	sxar2
/*    104 */	mulx	%xg5,%xg2,%xg5
/*    104 */	add	%xg7,%xg5,%xg7

/*    104 */	sxar1
/*    104 */	sub	%xg7,1,%xg7

/*    104 */	ba	.L6934
	nop


.L6931:

/*    104 */	sxar1
/*    104 */	cmp	%xg5,%xg3

/*    104 */	bl	.L6933
	nop


.L6932:


/*    104 */	sxar2
/*    104 */	mulx	%xg5,%xg2,%xg5
/*    104 */	add	%xg2,%xg2,%xg7


/*    104 */	sxar2
/*    104 */	add	%xg5,%xg3,%xg5
/*    104 */	add	%xg5,%xg5,%xg5


/*    104 */	sxar2
/*    104 */	add	%xg7,%xg5,%xg7
/*    104 */	sub	%xg7,1,%xg7

/*    104 */	ba	.L6934
	nop


.L6933:


/*    104 */	sxar2
/*    104 */	add	%xg2,1,%xg2
/*    104 */	add	%xg5,%xg5,%xg5


/*    104 */	sxar2
/*    104 */	mulx	%xg5,%xg2,%xg5
/*    104 */	add	%xg2,%xg2,%xg7


/*    104 */	sxar2
/*    104 */	add	%xg7,%xg5,%xg7
/*    104 */	sub	%xg7,1,%xg7

.L6934:

/*    104 */	sxar1
/*    104 */	cmp	%xg2,%g0

/*    104 */	be	.L6939
	nop


.L6935:


/*    104 */	sxar2
/*    104 */	sub	%xg7,%xg5,%xg7
/*    104 */	ldd,s	[%i2+-3680],%f164


/*    104 */	sxar2
/*    104 */	ldd,s	[%i2+-3664],%f222
/*    104 */	srl	%xg7,31,%xg8

/*    104 */	sxar1
/*    104 */	ldd,s	[%i2+-3712],%f220

/*    104 */	ldx	[%i0+2175],%g5


/*    104 */	sxar2
/*    104 */	add	%xg7,%xg8,%xg7
/*    104 */	ldd,s	[%i2+-3696],%f218


/*    104 */	sxar2
/*    104 */	sra	%xg7,1,%xg7
/*    104 */	add	%xg7,1,%xg7


/*    104 */	sxar2
/*    104 */	sra	%xg7,%g0,%xg7
/*    104 */	ldx	[%g5+16],%xg10


/*    104 */	sxar2
/*    104 */	ldd,s	[%i2+-3744],%f214
/*    104 */	sub	%i3,%xg7,%xg7


/*    104 */	sxar2
/*    104 */	ldd,s	[%i2+-3728],%f216
/*    104 */	ldd,s	[%i2+-3776],%f182


/*    104 */	sxar2
/*    104 */	srax	%xg7,32,%xg9
/*    104 */	ldd,s	[%i2+-3760],%f184


/*    104 */	sxar2
/*    104 */	ldd,s	[%i2+-3808],%f178
/*    104 */	and	%xg7,%xg9,%xg7


/*    104 */	sxar2
/*    104 */	ldd,s	[%i2+-3792],%f180
/*    104 */	ldd,s	[%i2+-3840],%f174


/*    195 */	sxar2
/*    195 */	sub	%i1,%xg7,%xg7
/*    195 */	add	%xg10,48,%xg11


/*    104 */	sxar2
/*    104 */	ldd,s	[%i2+-3824],%f176
/*    104 */	ldd,s	[%i2+-3872],%f160


/*    195 */	sxar2
/*    195 */	cmp	%xg7,7
/*    195 */	add	%xg10,16,%xg12


/*    104 */	sxar2
/*    104 */	ldd,s	[%i2+-3856],%f158
/*    104 */	ldd,s	[%i2+-3904],%f156


/*    195 */	sxar2
/*    195 */	add	%xg10,32,%xg13
/*    195 */	add	%xg10,64,%xg14


/*    104 */	sxar2
/*    104 */	ldd,s	[%i2+-3888],%f154
/*    104 */	ldd,s	[%i2+-3936],%f152


/*    195 */	sxar2
/*    195 */	add	%xg10,80,%xg15
/*    195 */	add	%xg10,96,%xg16


/*    104 */	sxar2
/*    104 */	ldd,s	[%i2+-3920],%f150
/*    104 */	ldd,s	[%i2+-3968],%f64


/*    195 */	sxar2
/*    195 */	add	%xg10,112,%xg17
/*    195 */	add	%xg10,128,%xg18


/*    104 */	sxar2
/*    104 */	ldd,s	[%i2+-3952],%f60
/*    104 */	ldd,s	[%i2+-4000],%f56


/*    195 */	sxar2
/*    195 */	add	%xg10,144,%xg19
/*    195 */	add	%xg10,160,%xg20


/*    104 */	sxar2
/*    104 */	ldd,s	[%i2+-3984],%f52
/*    104 */	ldd,s	[%i2+-4032],%f48


/*    195 */	sxar2
/*    195 */	add	%xg10,176,%xg21
/*    195 */	add	%xg10,192,%xg22


/*    104 */	sxar2
/*    104 */	ldd,s	[%i2+-4016],%f44
/*    104 */	ldd,s	[%i2+-3648],%f192


/*    104 */	sxar2
/*    104 */	ldd,s	[%i2+-3632],%f194
/*    104 */	ldd,s	[%i2+-3616],%f196


/*    104 */	sxar2
/*    104 */	ldd,s	[%i2+-3600],%f198
/*    104 */	ldd,s	[%i2+-3584],%f204


/*    104 */	sxar2
/*    104 */	ldd,s	[%i2+-3568],%f208
/*    104 */	ldd,s	[%i2+-3552],%f212


/*    104 */	sxar2
/*    104 */	ldd,s	[%i2+-3536],%f188
/*    104 */	ldd,s	[%i2+-3520],%f190


/*    104 */	sxar2
/*    104 */	ldd,s	[%i2+-3504],%f202
/*    104 */	ldd,s	[%i2+-3488],%f200


/*    104 */	sxar2
/*    104 */	ldd,s	[%i2+-3472],%f206
/*    104 */	ldd,s	[%i2+-3456],%f210

/*    104 */	bl	.L7085
	nop


.L7081:


.L7088:


/*    104 */	sxar2
/*    104 */	srl	%xg5,31,%xg23
/*    ??? */	sethi	%hi(8160),%xg31


/*    104 */	sxar2
/*    104 */	add	%xg23,%xg5,%xg23
/*    ??? */	xor	%xg31,-993,%xg31


/*    104 */	sxar2
/*    104 */	add	%xg5,2,%xg24
/*    104 */	sra	%xg23,1,%xg23


/*    104 */	sxar2
/*    104 */	srl	%xg24,31,%xg25
/*    104 */	sra	%xg23,%g0,%xg23


/*    104 */	sxar2
/*    104 */	add	%xg25,%xg24,%xg25
/*    104 */	mulx	%xg23,208,%xg23

/*    104 */	sxar1
/*    104 */	sra	%xg25,1,%xg25

/*    ??? */	sethi	%hi(8176),%g1

/*    104 */	sxar1
/*    104 */	sra	%xg25,%g0,%xg25

/*    ??? */	xor	%g1,-1009,%g1

/*    104 */	sxar1
/*    ??? */	ldd,s	[%fp+%g1],%f244

/*    ??? */	sethi	%hi(8208),%g2

/*    ??? */	sethi	%hi(8224),%g3

/*    ??? */	xor	%g2,-17,%g2

/*    ??? */	sethi	%hi(8192),%g4

/*    104 */	sxar1
/*    ??? */	ldd,s	[%fp+%g2],%f74

/*    ??? */	xor	%g3,-33,%g3

/*    ??? */	xor	%g4,-1,%g4


/*    104 */	sxar2
/*    104 */	add	%xg5,4,%xg5
/*    104 */	ldd,s	[%xg23+%xg10],%f46


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg23+%xg18],%f128
/*    104 */	srl	%xg5,31,%xg26


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg23+%xg12],%f54
/*    ??? */	ldd,s	[%fp+%xg31],%f242


/*    104 */	sxar2
/*    104 */	add	%xg26,%xg5,%xg26
/*    104 */	ldd,s	[%xg23+%xg13],%f62


/*    104 */	sxar2
/*    104 */	mulx	%xg25,208,%xg25
/*    104 */	ldd,s	[%xg23+%xg15],%f140


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg23+%xg14],%f136
/*    104 */	ldd,s	[%xg23+%xg16],%f144


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg23+%xg17],%f148
/*    104 */	fmsubd,sc	%f46,%f242,%f192,%f50


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f302,%f242,%f192,%f46
/*    104 */	ldd,s	[%xg23+%xg19],%f240


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f54,%f242,%f194,%f58
/*    104 */	fmsubd,sc	%f310,%f242,%f194,%f54


/*    104 */	sxar2
/*    ??? */	ldd,s	[%fp+%g3],%f72
/*    104 */	fmsubd,sc	%f62,%f242,%f196,%f66


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f318,%f242,%f196,%f62
/*    ??? */	ldd,s	[%fp+%g4],%f70


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg25+%xg10],%f34
/*    104 */	ldd,s	[%xg25+%xg12],%f112


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f140,%f242,%f204,%f138
/*    104 */	fmsubd,sc	%f396,%f242,%f204,%f140


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg25+%xg13],%f38
/*    104 */	fmsubd,sc	%f136,%f242,%f198,%f134


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f144,%f242,%f208,%f142
/*    104 */	fmsubd,sc	%f392,%f242,%f198,%f136


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f148,%f242,%f212,%f146
/*    104 */	fmaddd,s	%f50,%f50,%f210,%f234


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f46,%f46,%f210,%f236
/*    104 */	fmsubd,sc	%f34,%f242,%f192,%f32


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f290,%f242,%f192,%f34
/*    104 */	fmsubd,sc	%f112,%f242,%f194,%f114


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f368,%f242,%f194,%f112
/*    104 */	fmsubd,sc	%f38,%f242,%f196,%f36


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f294,%f242,%f196,%f38
/*    104 */	fmsubd,sc	%f400,%f242,%f208,%f144


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f404,%f242,%f212,%f148
/*    104 */	fmuld,s	%f58,%f138,%f40


/*    104 */	sxar2
/*    104 */	fmuld,s	%f54,%f140,%f42
/*    104 */	fmaddd,s	%f58,%f58,%f234,%f234


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f54,%f54,%f236,%f236
/*    104 */	fmaddd,s	%f32,%f32,%f210,%f130


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f34,%f34,%f210,%f132
/*    104 */	fmaddd,s	%f66,%f66,%f234,%f234


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f62,%f62,%f236,%f236
/*    104 */	fmaddd,s	%f114,%f114,%f130,%f130


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f112,%f112,%f132,%f132
/*    104 */	frsqrtad,s	%f234,%f186


/*    104 */	sxar2
/*    104 */	frsqrtad,s	%f236,%f238
/*    104 */	fmuld,s	%f234,%f244,%f234


/*    104 */	sxar2
/*    104 */	fmuld,s	%f236,%f244,%f236
/*    104 */	fmaddd,s	%f36,%f36,%f130,%f130


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f38,%f38,%f132,%f132
/*    104 */	fmuld,s	%f186,%f186,%f162


/*    104 */	sxar2
/*    104 */	fmuld,s	%f238,%f238,%f166
/*    104 */	fnmsubd,s	%f234,%f162,%f244,%f162


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f236,%f166,%f244,%f166
/*    104 */	fmaddd,s	%f186,%f162,%f186,%f186


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f238,%f166,%f238,%f238
/*    104 */	fmuld,s	%f186,%f186,%f168


/*    104 */	sxar2
/*    104 */	fmuld,s	%f238,%f238,%f170
/*    104 */	fnmsubd,s	%f234,%f168,%f244,%f168


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f236,%f170,%f244,%f170
/*    104 */	fmaddd,s	%f186,%f168,%f186,%f186


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f238,%f170,%f238,%f238
/*    104 */	fmuld,s	%f186,%f186,%f172


/*    104 */	sxar2
/*    104 */	fmuld,s	%f238,%f238,%f224
/*    104 */	fnmsubd,s	%f234,%f172,%f244,%f234

/*    104 */	sxar1
/*    104 */	fnmsubd,s	%f236,%f224,%f244,%f236

.L6936:


/*    104 */	sxar2
/*    104 */	sra	%xg26,1,%xg26
/*    104 */	fmsubd,sc	%f128,%f242,%f188,%f76


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f384,%f242,%f188,%f128
/*    104 */	sra	%xg26,%g0,%xg26


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f240,%f242,%f190,%f94
/*    104 */	frsqrtad,s	%f130,%f162


/*    104 */	sxar2
/*    104 */	mulx	%xg26,208,%xg26
/*    104 */	fmsubd,sc	%f496,%f242,%f190,%f240


/*    104 */	sxar2
/*    104 */	frsqrtad,s	%f132,%f84
/*    104 */	fmaddd,s	%f50,%f134,%f40,%f40


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f46,%f136,%f42,%f42
/*    104 */	fmuld,s	%f138,%f138,%f80


/*    104 */	sxar2
/*    104 */	fmuld,s	%f140,%f140,%f78
/*    104 */	fmaddd,s	%f186,%f234,%f186,%f186


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f238,%f236,%f238,%f238
/*    104 */	fmuld,s	%f58,%f76,%f226


/*    104 */	sxar2
/*    104 */	fmuld,s	%f54,%f128,%f228
/*    104 */	fmuld,s	%f130,%f244,%f130


/*    104 */	sxar2
/*    104 */	fmuld,s	%f162,%f162,%f230
/*    104 */	ldd,s	[%xg26+%xg10],%f248


/*    104 */	sxar2
/*    104 */	fmuld,s	%f132,%f244,%f132
/*    104 */	fmuld,s	%f84,%f84,%f232


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f66,%f142,%f40,%f40
/*    104 */	fmaddd,s	%f62,%f144,%f42,%f42


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg26+%xg12],%f252
/*    104 */	fmaddd,s	%f134,%f134,%f80,%f80


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f136,%f136,%f78,%f78
/*    104 */	fmuld,s	%f186,%f186,%f108


/*    104 */	sxar2
/*    104 */	fmuld,s	%f238,%f238,%f110
/*    104 */	fmaddd,s	%f50,%f146,%f226,%f226


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f46,%f148,%f228,%f228
/*    104 */	fmsubd,sc	%f248,%f242,%f192,%f246


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f504,%f242,%f192,%f248
/*    104 */	ldd,s	[%xg26+%xg13],%f68


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f130,%f230,%f244,%f230
/*    104 */	fnmsubd,s	%f132,%f232,%f244,%f232


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f252,%f242,%f194,%f250
/*    104 */	fmsubd,sc	%f508,%f242,%f194,%f252


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg25+%xg14],%f170
/*    104 */	fmaddd,s	%f142,%f142,%f80,%f80


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f144,%f144,%f78,%f78
/*    104 */	ldd,s	[%xg23+%xg21],%f126


/*    104 */	sxar2
/*    104 */	fmuld,s	%f40,%f108,%f40
/*    104 */	fmuld,s	%f42,%f110,%f42


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f66,%f94,%f226,%f226
/*    104 */	fmaddd,s	%f62,%f240,%f228,%f228


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f68,%f242,%f196,%f254
/*    104 */	fmsubd,sc	%f324,%f242,%f196,%f68


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg23+%xg20],%f122
/*    104 */	fmaddd,s	%f162,%f230,%f162,%f162


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f84,%f232,%f84,%f84
/*    104 */	fmsubd,sc	%f170,%f242,%f198,%f168


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f426,%f242,%f198,%f170
/*    104 */	fmsubd,sc	%f126,%f242,%f200,%f234


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f382,%f242,%f200,%f126
/*    104 */	fmuld,s	%f40,%f40,%f116


/*    104 */	sxar2
/*    104 */	fmuld,s	%f42,%f42,%f118
/*    104 */	ldd,s	[%xg25+%xg15],%f224


/*    104 */	sxar2
/*    104 */	faddd,s	%f80,%f226,%f80
/*    104 */	faddd,s	%f78,%f228,%f78


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f122,%f242,%f202,%f124
/*    104 */	fmsubd,sc	%f378,%f242,%f202,%f122


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg23+%xg22],%f102
/*    104 */	fmuld,s	%f162,%f162,%f120


/*    104 */	sxar2
/*    104 */	fmuld,s	%f84,%f84,%f230
/*    104 */	fmuld,s	%f138,%f76,%f90


/*    104 */	sxar2
/*    104 */	fmuld,s	%f140,%f128,%f88
/*    104 */	fmuld,s	%f58,%f234,%f236


/*    104 */	sxar2
/*    104 */	fmuld,s	%f54,%f126,%f92
/*    104 */	fmsubd,sc	%f224,%f242,%f204,%f172


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f480,%f242,%f204,%f224
/*    104 */	ldd,s	[%xg25+%xg16],%f228


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f108,%f80,%f116,%f80
/*    104 */	fmaddd,s	%f110,%f78,%f118,%f78


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f102,%f242,%f206,%f98
/*    104 */	fmsubd,sc	%f358,%f242,%f206,%f102


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f130,%f120,%f244,%f120
/*    104 */	fnmsubd,s	%f132,%f230,%f244,%f230


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f134,%f146,%f90,%f90
/*    104 */	fmaddd,s	%f136,%f148,%f88,%f88


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f50,%f124,%f236,%f236
/*    104 */	fmaddd,s	%f46,%f122,%f92,%f92


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f228,%f242,%f208,%f226
/*    104 */	fmsubd,sc	%f484,%f242,%f208,%f228


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg25+%xg17],%f232
/*    104 */	fmuld,s	%f70,%f80,%f80


/*    104 */	sxar2
/*    104 */	fmuld,s	%f70,%f78,%f78
/*    104 */	fmaddd,s	%f246,%f246,%f210,%f166


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f248,%f248,%f210,%f86
/*    104 */	fmaddd,s	%f162,%f120,%f162,%f162


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f84,%f230,%f84,%f84
/*    104 */	fmaddd,s	%f142,%f94,%f90,%f90


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f144,%f240,%f88,%f88
/*    104 */	fmaddd,s	%f66,%f98,%f236,%f236


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f62,%f102,%f92,%f92
/*    104 */	ldd,s	[%xg23+%xg11],%f82


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f232,%f242,%f212,%f230
/*    104 */	fmsubd,sc	%f488,%f242,%f212,%f232


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f72,%f116,%f80,%f116
/*    104 */	fnmsubd,s	%f72,%f118,%f78,%f118


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f250,%f250,%f166,%f166
/*    104 */	fmaddd,s	%f252,%f252,%f86,%f86


/*    104 */	sxar2
/*    104 */	fmuld,s	%f162,%f162,%f104
/*    104 */	fmuld,s	%f84,%f84,%f106


/*    104 */	sxar2
/*    104 */	fmaddd,sc	%f82,%f186,%f74,%f186
/*    104 */	fmaddd,sc	%f338,%f238,%f74,%f82


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f70,%f90,%f236,%f90
/*    104 */	fmaddd,s	%f70,%f88,%f92,%f88


/*    104 */	sxar2
/*    104 */	fmuld,s	%f70,%f40,%f92
/*    104 */	fmuld,s	%f70,%f42,%f236


/*    104 */	sxar2
/*    104 */	fmuld,s	%f40,%f116,%f40
/*    104 */	fmuld,s	%f42,%f118,%f42


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f254,%f254,%f166,%f166
/*    104 */	fmaddd,s	%f68,%f68,%f86,%f86


/*    104 */	sxar2
/*    104 */	fmuld,s	%f114,%f172,%f96
/*    104 */	fmuld,s	%f112,%f224,%f100


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f130,%f104,%f244,%f130
/*    104 */	fnmsubd,s	%f132,%f106,%f244,%f132


/*    104 */	sxar2
/*    104 */	fmuld,s	%f186,%f108,%f186
/*    104 */	fmuld,s	%f82,%f110,%f82


/*    104 */	sxar2
/*    104 */	faddd,s	%f92,%f92,%f104
/*    104 */	faddd,s	%f236,%f236,%f106


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f90,%f108,%f40,%f90
/*    104 */	fmaddd,s	%f88,%f110,%f42,%f88


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f92,%f50,%f134,%f134
/*    104 */	fnmsubd,s	%f236,%f46,%f136,%f136


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f92,%f58,%f138,%f138
/*    104 */	fnmsubd,s	%f236,%f54,%f140,%f140


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f92,%f66,%f142,%f142
/*    104 */	fnmsubd,s	%f236,%f62,%f144,%f144


/*    104 */	sxar2
/*    104 */	fmuld,s	%f70,%f92,%f92
/*    104 */	fmuld,s	%f70,%f236,%f236


/*    104 */	sxar2
/*    104 */	fmuld,s	%f70,%f80,%f110
/*    104 */	fmuld,s	%f70,%f78,%f108


/*    104 */	sxar2
/*    104 */	fmuld,s	%f70,%f90,%f90
/*    104 */	fmuld,s	%f70,%f88,%f88


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f104,%f134,%f146,%f146
/*    104 */	fnmsubd,s	%f106,%f136,%f148,%f148


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f104,%f138,%f76,%f76
/*    104 */	fnmsubd,s	%f106,%f140,%f128,%f128


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f104,%f142,%f94,%f104
/*    104 */	fnmsubd,s	%f106,%f144,%f240,%f106


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f82,%f46,%f44,%f44
/*    104 */	fmaddd,s	%f186,%f50,%f48,%f48


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f82,%f54,%f52,%f52
/*    104 */	fmaddd,s	%f186,%f58,%f56,%f56


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f82,%f62,%f60,%f60
/*    104 */	fmaddd,s	%f186,%f66,%f64,%f64


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f80,%f50,%f146,%f146
/*    104 */	fnmsubd,s	%f78,%f46,%f148,%f148


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f80,%f58,%f76,%f76
/*    104 */	fnmsubd,s	%f78,%f54,%f128,%f128


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f80,%f66,%f104,%f80
/*    104 */	fnmsubd,s	%f78,%f62,%f106,%f78


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f236,%f148,%f122,%f122
/*    104 */	fnmsubd,s	%f92,%f146,%f124,%f124


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f236,%f128,%f126,%f126
/*    104 */	fnmsubd,s	%f92,%f76,%f234,%f234


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f236,%f78,%f102,%f236
/*    104 */	fnmsubd,s	%f92,%f80,%f98,%f92


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f82,%f136,%f150,%f150
/*    104 */	fmaddd,s	%f186,%f134,%f152,%f152


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f82,%f140,%f154,%f154
/*    104 */	fmaddd,s	%f186,%f138,%f156,%f156


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f82,%f144,%f158,%f158
/*    104 */	fmaddd,s	%f186,%f142,%f160,%f160


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f108,%f136,%f122,%f136
/*    104 */	fnmsubd,s	%f110,%f134,%f124,%f134


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f108,%f140,%f126,%f140
/*    104 */	fnmsubd,s	%f110,%f138,%f234,%f138


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f108,%f144,%f236,%f108
/*    104 */	fnmsubd,s	%f110,%f142,%f92,%f110


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f88,%f46,%f136,%f136
/*    104 */	fnmsubd,s	%f90,%f50,%f134,%f134


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f88,%f54,%f140,%f140
/*    104 */	fnmsubd,s	%f90,%f58,%f138,%f138


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f88,%f62,%f108,%f88
/*    104 */	fnmsubd,s	%f90,%f66,%f110,%f90


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f82,%f148,%f176,%f148
/*    104 */	fmaddd,s	%f186,%f146,%f174,%f146


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f82,%f128,%f180,%f128
/*    104 */	fmaddd,s	%f186,%f76,%f178,%f76


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f82,%f78,%f184,%f78
/*    104 */	fmaddd,s	%f186,%f80,%f182,%f80


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f82,%f136,%f216,%f136
/*    104 */	fmaddd,s	%f186,%f134,%f214,%f134


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f82,%f140,%f218,%f140
/*    104 */	fmaddd,s	%f186,%f138,%f220,%f138


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f82,%f88,%f222,%f82
/*    104 */	fmaddd,s	%f186,%f90,%f164,%f186


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg25+%xg18],%f88
/*    104 */	add	%xg5,2,%xg28


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg25+%xg19],%f178
/*    104 */	srl	%xg28,31,%xg23


/*    104 */	sxar2
/*    104 */	add	%xg23,%xg28,%xg23
/*    104 */	sra	%xg23,1,%xg23


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f88,%f242,%f188,%f90
/*    104 */	fmsubd,sc	%f344,%f242,%f188,%f88


/*    104 */	sxar2
/*    104 */	sra	%xg23,%g0,%xg23
/*    104 */	fmsubd,sc	%f178,%f242,%f190,%f144


/*    104 */	sxar2
/*    104 */	frsqrtad,s	%f166,%f116
/*    104 */	mulx	%xg23,208,%xg23


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f434,%f242,%f190,%f178
/*    104 */	frsqrtad,s	%f86,%f98


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f32,%f168,%f96,%f96
/*    104 */	fmaddd,s	%f34,%f170,%f100,%f100


/*    104 */	sxar2
/*    104 */	fmuld,s	%f172,%f172,%f94
/*    104 */	fmuld,s	%f224,%f224,%f92


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f162,%f130,%f162,%f162
/*    104 */	fmaddd,s	%f84,%f132,%f84,%f84


/*    104 */	sxar2
/*    104 */	fmuld,s	%f114,%f90,%f108
/*    104 */	fmuld,s	%f112,%f88,%f110


/*    104 */	sxar2
/*    104 */	fmuld,s	%f166,%f244,%f164
/*    104 */	fmuld,s	%f116,%f116,%f120


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg23+%xg10],%f46
/*    104 */	fmuld,s	%f86,%f244,%f86


/*    104 */	sxar2
/*    104 */	fmuld,s	%f98,%f98,%f240
/*    104 */	fmaddd,s	%f36,%f226,%f96,%f96


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f38,%f228,%f100,%f100
/*    104 */	ldd,s	[%xg23+%xg12],%f54


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f168,%f168,%f94,%f94
/*    104 */	fmaddd,s	%f170,%f170,%f92,%f92


/*    104 */	sxar2
/*    104 */	fmuld,s	%f162,%f162,%f166
/*    104 */	fmuld,s	%f84,%f84,%f238


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f32,%f230,%f108,%f108
/*    104 */	fmaddd,s	%f34,%f232,%f110,%f110


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f46,%f242,%f192,%f50
/*    104 */	fmsubd,sc	%f302,%f242,%f192,%f46


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg23+%xg13],%f62
/*    104 */	fnmsubd,s	%f164,%f120,%f244,%f120


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f86,%f240,%f244,%f240
/*    104 */	fmsubd,sc	%f54,%f242,%f194,%f58


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f310,%f242,%f194,%f54
/*    104 */	ldd,s	[%xg26+%xg14],%f216


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f226,%f226,%f94,%f94
/*    104 */	fmaddd,s	%f228,%f228,%f92,%f92


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg25+%xg21],%f122
/*    104 */	fmuld,s	%f96,%f166,%f96


/*    104 */	sxar2
/*    104 */	fmuld,s	%f100,%f238,%f100
/*    104 */	fmaddd,s	%f36,%f144,%f108,%f108


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f38,%f178,%f110,%f110
/*    104 */	fmsubd,sc	%f62,%f242,%f196,%f66


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f318,%f242,%f196,%f62
/*    104 */	ldd,s	[%xg25+%xg20],%f118


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f116,%f120,%f116,%f116
/*    104 */	fmaddd,s	%f98,%f240,%f98,%f98


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f216,%f242,%f198,%f214
/*    104 */	fmsubd,sc	%f472,%f242,%f198,%f216


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f122,%f242,%f200,%f124
/*    104 */	fmsubd,sc	%f378,%f242,%f200,%f122


/*    104 */	sxar2
/*    104 */	fmuld,s	%f96,%f96,%f40
/*    104 */	fmuld,s	%f100,%f100,%f42


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg26+%xg15],%f218
/*    104 */	faddd,s	%f94,%f108,%f94


/*    104 */	sxar2
/*    104 */	faddd,s	%f92,%f110,%f92
/*    104 */	fmsubd,sc	%f118,%f242,%f202,%f120


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f374,%f242,%f202,%f118
/*    104 */	ldd,s	[%xg25+%xg22],%f182


/*    104 */	sxar2
/*    104 */	fmuld,s	%f116,%f116,%f102
/*    104 */	fmuld,s	%f98,%f98,%f104


/*    104 */	sxar2
/*    104 */	fmuld,s	%f172,%f90,%f132
/*    104 */	fmuld,s	%f224,%f88,%f130


/*    104 */	sxar2
/*    104 */	fmuld,s	%f114,%f124,%f110
/*    104 */	fmuld,s	%f112,%f122,%f126


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f218,%f242,%f204,%f220
/*    104 */	fmsubd,sc	%f474,%f242,%f204,%f218


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg26+%xg16],%f108
/*    104 */	fmaddd,s	%f166,%f94,%f40,%f94


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f238,%f92,%f42,%f92
/*    104 */	fmsubd,sc	%f182,%f242,%f206,%f180


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f438,%f242,%f206,%f182
/*    104 */	fnmsubd,s	%f164,%f102,%f244,%f102


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f86,%f104,%f244,%f104
/*    104 */	fmaddd,s	%f168,%f230,%f132,%f132


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f170,%f232,%f130,%f130
/*    104 */	fmaddd,s	%f32,%f120,%f110,%f110


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f34,%f118,%f126,%f126
/*    104 */	fmsubd,sc	%f108,%f242,%f208,%f106


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f364,%f242,%f208,%f108
/*    104 */	ldd,s	[%xg26+%xg17],%f176


/*    104 */	sxar2
/*    104 */	fmuld,s	%f70,%f94,%f94
/*    104 */	fmuld,s	%f70,%f92,%f92


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f50,%f50,%f210,%f234
/*    104 */	fmaddd,s	%f46,%f46,%f210,%f236


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f116,%f102,%f116,%f116
/*    104 */	fmaddd,s	%f98,%f104,%f98,%f98


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f226,%f144,%f132,%f132
/*    104 */	fmaddd,s	%f228,%f178,%f130,%f130


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f36,%f180,%f110,%f110
/*    104 */	fmaddd,s	%f38,%f182,%f126,%f126


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg25+%xg11],%f102
/*    104 */	fmsubd,sc	%f176,%f242,%f212,%f174


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f432,%f242,%f212,%f176
/*    104 */	fnmsubd,s	%f72,%f40,%f94,%f40


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f72,%f42,%f92,%f42
/*    104 */	fmaddd,s	%f58,%f58,%f234,%f234


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f54,%f54,%f236,%f236
/*    104 */	fmuld,s	%f116,%f116,%f184


/*    104 */	sxar2
/*    104 */	fmuld,s	%f98,%f98,%f222
/*    104 */	fmaddd,sc	%f102,%f162,%f74,%f162


/*    104 */	sxar2
/*    104 */	fmaddd,sc	%f358,%f84,%f74,%f102
/*    104 */	fmaddd,s	%f70,%f132,%f110,%f132


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f70,%f130,%f126,%f130
/*    104 */	fmuld,s	%f70,%f96,%f142


/*    104 */	sxar2
/*    104 */	fmuld,s	%f70,%f100,%f126
/*    104 */	fmuld,s	%f96,%f40,%f96


/*    104 */	sxar2
/*    104 */	fmuld,s	%f100,%f42,%f100
/*    104 */	fmaddd,s	%f66,%f66,%f234,%f234


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f62,%f62,%f236,%f236
/*    104 */	fmuld,s	%f250,%f220,%f104


/*    104 */	sxar2
/*    104 */	fmuld,s	%f252,%f218,%f110
/*    104 */	fnmsubd,s	%f164,%f184,%f244,%f164


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f86,%f222,%f244,%f86
/*    104 */	fmuld,s	%f162,%f166,%f162


/*    104 */	sxar2
/*    104 */	fmuld,s	%f102,%f238,%f102
/*    104 */	faddd,s	%f142,%f142,%f184


/*    104 */	sxar2
/*    104 */	faddd,s	%f126,%f126,%f222
/*    104 */	fmaddd,s	%f132,%f166,%f96,%f132


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f130,%f238,%f100,%f130
/*    104 */	fnmsubd,s	%f142,%f32,%f168,%f168


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f126,%f34,%f170,%f170
/*    104 */	fnmsubd,s	%f142,%f114,%f172,%f172


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f126,%f112,%f224,%f224
/*    104 */	fnmsubd,s	%f142,%f36,%f226,%f226


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f126,%f38,%f228,%f228
/*    104 */	fmuld,s	%f70,%f142,%f142


/*    104 */	sxar2
/*    104 */	fmuld,s	%f70,%f126,%f126
/*    104 */	fmuld,s	%f70,%f94,%f240


/*    104 */	sxar2
/*    104 */	fmuld,s	%f70,%f92,%f238
/*    104 */	fmuld,s	%f70,%f132,%f132


/*    104 */	sxar2
/*    104 */	fmuld,s	%f70,%f130,%f130
/*    104 */	fnmsubd,s	%f184,%f168,%f230,%f230


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f222,%f170,%f232,%f232
/*    104 */	fnmsubd,s	%f184,%f172,%f90,%f90


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f222,%f224,%f88,%f88
/*    104 */	fnmsubd,s	%f184,%f226,%f144,%f184


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f222,%f228,%f178,%f222
/*    104 */	fmaddd,s	%f102,%f34,%f44,%f44


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f162,%f32,%f48,%f48
/*    104 */	fmaddd,s	%f102,%f112,%f52,%f52


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f162,%f114,%f56,%f56
/*    104 */	fmaddd,s	%f102,%f38,%f60,%f60


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f162,%f36,%f64,%f64
/*    104 */	fnmsubd,s	%f94,%f32,%f230,%f230


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f92,%f34,%f232,%f232
/*    104 */	fnmsubd,s	%f94,%f114,%f90,%f90


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f92,%f112,%f88,%f88
/*    104 */	fnmsubd,s	%f94,%f36,%f184,%f94


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f92,%f38,%f222,%f92
/*    104 */	fnmsubd,s	%f126,%f232,%f118,%f118


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f142,%f230,%f120,%f120
/*    104 */	fnmsubd,s	%f126,%f88,%f122,%f122


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f142,%f90,%f124,%f124
/*    104 */	fnmsubd,s	%f126,%f92,%f182,%f126


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f142,%f94,%f180,%f142
/*    104 */	fmaddd,s	%f102,%f170,%f150,%f150


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f162,%f168,%f152,%f152
/*    104 */	fmaddd,s	%f102,%f224,%f154,%f154


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f162,%f172,%f156,%f156
/*    104 */	fmaddd,s	%f102,%f228,%f158,%f158


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f162,%f226,%f160,%f160
/*    104 */	fnmsubd,s	%f238,%f170,%f118,%f170


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f240,%f168,%f120,%f168
/*    104 */	fnmsubd,s	%f238,%f224,%f122,%f224


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f240,%f172,%f124,%f172
/*    104 */	fnmsubd,s	%f238,%f228,%f126,%f238


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f240,%f226,%f142,%f240
/*    104 */	fnmsubd,s	%f130,%f34,%f170,%f170


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f132,%f32,%f168,%f168
/*    104 */	fnmsubd,s	%f130,%f112,%f224,%f224


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f132,%f114,%f172,%f172
/*    104 */	fnmsubd,s	%f130,%f38,%f238,%f130


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f132,%f36,%f240,%f132
/*    104 */	fmaddd,s	%f102,%f232,%f148,%f232


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f162,%f230,%f146,%f230
/*    104 */	fmaddd,s	%f102,%f88,%f128,%f88


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f162,%f90,%f76,%f90
/*    104 */	fmaddd,s	%f102,%f92,%f78,%f92


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f162,%f94,%f80,%f94
/*    104 */	fmaddd,s	%f102,%f170,%f136,%f170


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f162,%f168,%f134,%f168
/*    104 */	fmaddd,s	%f102,%f224,%f140,%f224


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f162,%f172,%f138,%f172
/*    104 */	fmaddd,s	%f102,%f130,%f82,%f102


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f162,%f132,%f186,%f162
/*    104 */	ldd,s	[%xg26+%xg18],%f180


/*    104 */	sxar2
/*    104 */	add	%xg5,4,%xg29
/*    104 */	ldd,s	[%xg26+%xg19],%f78


/*    104 */	sxar2
/*    104 */	srl	%xg29,31,%xg25
/*    104 */	add	%xg25,%xg29,%xg25


/*    104 */	sxar2
/*    104 */	sra	%xg25,1,%xg25
/*    104 */	fmsubd,sc	%f180,%f242,%f188,%f178


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f436,%f242,%f188,%f180
/*    104 */	sra	%xg25,%g0,%xg25


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f78,%f242,%f190,%f76
/*    104 */	frsqrtad,s	%f234,%f186


/*    104 */	sxar2
/*    104 */	mulx	%xg25,208,%xg25
/*    104 */	fmsubd,sc	%f334,%f242,%f190,%f78


/*    104 */	sxar2
/*    104 */	frsqrtad,s	%f236,%f238
/*    104 */	fmaddd,s	%f246,%f214,%f104,%f104


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f248,%f216,%f110,%f110
/*    104 */	fmuld,s	%f220,%f220,%f182


/*    104 */	sxar2
/*    104 */	fmuld,s	%f218,%f218,%f184
/*    104 */	fmaddd,s	%f116,%f164,%f116,%f164


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f98,%f86,%f98,%f98
/*    104 */	fmuld,s	%f250,%f178,%f226


/*    104 */	sxar2
/*    104 */	fmuld,s	%f252,%f180,%f228
/*    104 */	fmuld,s	%f234,%f244,%f234


/*    104 */	sxar2
/*    104 */	fmuld,s	%f186,%f186,%f128
/*    104 */	ldd,s	[%xg25+%xg10],%f34


/*    104 */	sxar2
/*    104 */	fmuld,s	%f236,%f244,%f236
/*    104 */	fmuld,s	%f238,%f238,%f130


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f254,%f106,%f104,%f104
/*    104 */	fmaddd,s	%f68,%f108,%f110,%f110


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg25+%xg12],%f112
/*    104 */	fmaddd,s	%f214,%f214,%f182,%f182


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f216,%f216,%f184,%f184
/*    104 */	fmuld,s	%f164,%f164,%f240


/*    104 */	sxar2
/*    104 */	fmuld,s	%f98,%f98,%f126
/*    104 */	fmaddd,s	%f246,%f174,%f226,%f226


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f248,%f176,%f228,%f228
/*    104 */	fmsubd,sc	%f34,%f242,%f192,%f32


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f290,%f242,%f192,%f34
/*    104 */	ldd,s	[%xg25+%xg13],%f38


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f234,%f128,%f244,%f128
/*    104 */	fnmsubd,s	%f236,%f130,%f244,%f130


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f112,%f242,%f194,%f114
/*    104 */	fmsubd,sc	%f368,%f242,%f194,%f112


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg23+%xg14],%f136
/*    104 */	fmaddd,s	%f106,%f106,%f182,%f182


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f108,%f108,%f184,%f184
/*    104 */	ldd,s	[%xg26+%xg21],%f84


/*    104 */	sxar2
/*    104 */	fmuld,s	%f104,%f240,%f104
/*    104 */	fmuld,s	%f110,%f126,%f110


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f254,%f76,%f226,%f226
/*    104 */	fmaddd,s	%f68,%f78,%f228,%f228


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f38,%f242,%f196,%f36
/*    104 */	fmsubd,sc	%f294,%f242,%f196,%f38


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg26+%xg20],%f80
/*    104 */	fmaddd,s	%f186,%f128,%f186,%f186


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f238,%f130,%f238,%f238
/*    104 */	fmsubd,sc	%f136,%f242,%f198,%f134


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f392,%f242,%f198,%f136
/*    104 */	fmsubd,sc	%f84,%f242,%f200,%f86


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f340,%f242,%f200,%f84
/*    104 */	fmuld,s	%f104,%f104,%f166


/*    104 */	sxar2
/*    104 */	fmuld,s	%f110,%f110,%f40
/*    104 */	ldd,s	[%xg23+%xg15],%f140


/*    104 */	sxar2
/*    104 */	faddd,s	%f182,%f226,%f182
/*    104 */	faddd,s	%f184,%f228,%f184


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f80,%f242,%f202,%f82
/*    104 */	fmsubd,sc	%f336,%f242,%f202,%f80


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg26+%xg22],%f116
/*    104 */	fmuld,s	%f186,%f186,%f42


/*    104 */	sxar2
/*    104 */	fmuld,s	%f238,%f238,%f96
/*    104 */	fmuld,s	%f220,%f178,%f120


/*    104 */	sxar2
/*    104 */	fmuld,s	%f218,%f180,%f118
/*    104 */	fmuld,s	%f250,%f86,%f122


/*    104 */	sxar2
/*    104 */	fmuld,s	%f252,%f84,%f124
/*    104 */	fmsubd,sc	%f140,%f242,%f204,%f138


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f396,%f242,%f204,%f140
/*    104 */	ldd,s	[%xg23+%xg16],%f144


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f240,%f182,%f166,%f182
/*    104 */	fmaddd,s	%f126,%f184,%f40,%f184


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f116,%f242,%f206,%f100
/*    104 */	fmsubd,sc	%f372,%f242,%f206,%f116


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f234,%f42,%f244,%f42
/*    104 */	fnmsubd,s	%f236,%f96,%f244,%f96


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f214,%f174,%f120,%f120
/*    104 */	fmaddd,s	%f216,%f176,%f118,%f118


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f246,%f82,%f122,%f122
/*    104 */	fmaddd,s	%f248,%f80,%f124,%f124


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f144,%f242,%f208,%f142
/*    104 */	fmsubd,sc	%f400,%f242,%f208,%f144


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg23+%xg17],%f148
/*    104 */	fmuld,s	%f70,%f182,%f182


/*    104 */	sxar2
/*    104 */	fmuld,s	%f70,%f184,%f184
/*    104 */	fmaddd,s	%f32,%f32,%f210,%f130


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f34,%f34,%f210,%f132
/*    104 */	fmaddd,s	%f186,%f42,%f186,%f186


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f238,%f96,%f238,%f238
/*    104 */	fmaddd,s	%f106,%f76,%f120,%f120


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f108,%f78,%f118,%f118
/*    104 */	fmaddd,s	%f254,%f100,%f122,%f122


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f68,%f116,%f124,%f124
/*    104 */	ldd,s	[%xg26+%xg11],%f222


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f148,%f242,%f212,%f146
/*    104 */	fmsubd,sc	%f404,%f242,%f212,%f148


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f72,%f166,%f182,%f166
/*    104 */	fnmsubd,s	%f72,%f40,%f184,%f40


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f114,%f114,%f130,%f130
/*    104 */	fmaddd,s	%f112,%f112,%f132,%f132


/*    104 */	sxar2
/*    104 */	fmuld,s	%f186,%f186,%f128
/*    104 */	fmuld,s	%f238,%f238,%f226


/*    104 */	sxar2
/*    104 */	fmaddd,sc	%f222,%f164,%f74,%f164
/*    104 */	fmaddd,sc	%f478,%f98,%f74,%f222


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f70,%f120,%f122,%f120
/*    104 */	fmaddd,s	%f70,%f118,%f124,%f118


/*    104 */	sxar2
/*    104 */	fmuld,s	%f70,%f104,%f98
/*    104 */	fmuld,s	%f70,%f110,%f96


/*    104 */	sxar2
/*    104 */	fmuld,s	%f104,%f166,%f104
/*    104 */	fmuld,s	%f110,%f40,%f110


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f36,%f36,%f130,%f130
/*    104 */	fmaddd,s	%f38,%f38,%f132,%f132


/*    104 */	sxar2
/*    104 */	fmuld,s	%f58,%f138,%f40
/*    104 */	fmuld,s	%f54,%f140,%f42


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f234,%f128,%f244,%f234
/*    104 */	fnmsubd,s	%f236,%f226,%f244,%f236


/*    104 */	sxar2
/*    104 */	fmuld,s	%f164,%f240,%f164
/*    104 */	fmuld,s	%f222,%f126,%f222


/*    104 */	sxar2
/*    104 */	faddd,s	%f98,%f98,%f122
/*    104 */	faddd,s	%f96,%f96,%f124


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f120,%f240,%f104,%f120
/*    104 */	fmaddd,s	%f118,%f126,%f110,%f118


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f98,%f246,%f214,%f214
/*    104 */	fnmsubd,s	%f96,%f248,%f216,%f216


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f98,%f250,%f220,%f220
/*    104 */	fnmsubd,s	%f96,%f252,%f218,%f218


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f98,%f254,%f106,%f106
/*    104 */	fnmsubd,s	%f96,%f68,%f108,%f108


/*    104 */	sxar2
/*    104 */	fmuld,s	%f70,%f98,%f98
/*    104 */	fmuld,s	%f70,%f96,%f96


/*    104 */	sxar2
/*    104 */	fmuld,s	%f70,%f182,%f128
/*    104 */	fmuld,s	%f70,%f184,%f126


/*    104 */	sxar2
/*    104 */	fmuld,s	%f70,%f120,%f120
/*    104 */	fmuld,s	%f70,%f118,%f118


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f122,%f214,%f174,%f174
/*    104 */	fnmsubd,s	%f124,%f216,%f176,%f176


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f122,%f220,%f178,%f178
/*    104 */	fnmsubd,s	%f124,%f218,%f180,%f180


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f122,%f106,%f76,%f122
/*    104 */	fnmsubd,s	%f124,%f108,%f78,%f124


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f222,%f248,%f44,%f44
/*    104 */	fmaddd,s	%f164,%f246,%f48,%f48


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f222,%f252,%f52,%f52
/*    104 */	fmaddd,s	%f164,%f250,%f56,%f56


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f222,%f68,%f60,%f60
/*    104 */	fmaddd,s	%f164,%f254,%f64,%f64


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f182,%f246,%f174,%f174
/*    104 */	fnmsubd,s	%f184,%f248,%f176,%f176


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f182,%f250,%f178,%f178
/*    104 */	fnmsubd,s	%f184,%f252,%f180,%f180


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f182,%f254,%f122,%f182
/*    104 */	fnmsubd,s	%f184,%f68,%f124,%f184


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f96,%f176,%f80,%f80
/*    104 */	fnmsubd,s	%f98,%f174,%f82,%f82


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f96,%f180,%f84,%f84
/*    104 */	fnmsubd,s	%f98,%f178,%f86,%f86


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f96,%f184,%f116,%f96
/*    104 */	fnmsubd,s	%f98,%f182,%f100,%f98


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f222,%f216,%f150,%f150
/*    104 */	fmaddd,s	%f164,%f214,%f152,%f152


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f222,%f218,%f154,%f154
/*    104 */	fmaddd,s	%f164,%f220,%f156,%f156


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f222,%f108,%f158,%f158
/*    104 */	fmaddd,s	%f164,%f106,%f160,%f160


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f126,%f216,%f80,%f216
/*    104 */	fnmsubd,s	%f128,%f214,%f82,%f214


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f126,%f218,%f84,%f218
/*    104 */	fnmsubd,s	%f128,%f220,%f86,%f220


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f126,%f108,%f96,%f126
/*    104 */	fnmsubd,s	%f128,%f106,%f98,%f128


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f118,%f248,%f216,%f216
/*    104 */	fnmsubd,s	%f120,%f246,%f214,%f214


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f118,%f252,%f218,%f218
/*    104 */	fnmsubd,s	%f120,%f250,%f220,%f220


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f118,%f68,%f126,%f118
/*    104 */	fnmsubd,s	%f120,%f254,%f128,%f120


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f222,%f176,%f232,%f176
/*    104 */	fmaddd,s	%f164,%f174,%f230,%f174


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f222,%f180,%f88,%f180
/*    104 */	fmaddd,s	%f164,%f178,%f90,%f178


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f222,%f184,%f92,%f184
/*    104 */	fmaddd,s	%f164,%f182,%f94,%f182


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f222,%f216,%f170,%f216
/*    104 */	fmaddd,s	%f164,%f214,%f168,%f214


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f222,%f218,%f224,%f218
/*    104 */	fmaddd,s	%f164,%f220,%f172,%f220


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f222,%f118,%f102,%f222
/*    104 */	fmaddd,s	%f164,%f120,%f162,%f164


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg23+%xg18],%f128
/*    104 */	add	%xg5,6,%xg5


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg23+%xg19],%f240
/*    104 */	srl	%xg5,31,%xg26


/*    104 */	sxar2
/*    104 */	add	%xg26,%xg5,%xg26
/*    104 */	sub	%xg7,3,%xg7

/*    104 */	sxar1
/*    104 */	cmp	%xg7,8

/*    104 */	bge,pt	%icc, .L6936
	nop


.L7089:

/*    104 */	sxar1
/*    104 */	frsqrtad,s	%f130,%f68

/*    ??? */	sethi	%hi(8176),%o0

/*    ??? */	sethi	%hi(8160),%o1


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg23+%xg21],%f242
/*    104 */	frsqrtad,s	%f132,%f70

/*    104 */	sxar1
/*    104 */	ldd,s	[%xg25+%xg15],%f90

/*    ??? */	xor	%o0,-1009,%o0

/*    ??? */	xor	%o1,-993,%o1


/*    104 */	sxar2
/*    104 */	sra	%xg26,1,%xg26
/*    ??? */	ldd,s	[%fp+%o0],%f0

/*    104 */	sxar1
/*    ??? */	ldd,s	[%fp+%o1],%f162

/*    ??? */	sethi	%hi(8240),%o2


/*    104 */	sxar2
/*    104 */	sra	%xg26,%g0,%xg26
/*    104 */	ldd,s	[%xg25+%xg14],%f86

/*    104 */	sxar1
/*    104 */	ldd,s	[%xg23+%xg20],%f230

/*    ??? */	xor	%o2,-49,%o2


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg25+%xg18],%f102
/*    104 */	fmaddd,s	%f238,%f236,%f238,%f238

/*    104 */	sxar1
/*    104 */	ldd,s	[%xg25+%xg16],%f94

/*    ??? */	sethi	%hi(8256),%o3


/*    104 */	sxar2
/*    104 */	mulx	%xg26,208,%xg26
/*    104 */	fmaddd,s	%f46,%f136,%f42,%f42

/*    104 */	sxar1
/*    104 */	ldd,s	[%xg25+%xg17],%f98

/*    ??? */	xor	%o3,-65,%o3


/*    104 */	sxar2
/*    104 */	fmuld,s	%f130,%f0,%f130
/*    104 */	fmsubd,sc	%f240,%f162,%f190,%f166

/*    ??? */	sethi	%hi(8272),%o4

/*    ??? */	sethi	%hi(8288),%o5


/*    104 */	sxar2
/*    104 */	fmuld,s	%f132,%f0,%f132
/*    104 */	fmuld,s	%f68,%f68,%f84

/*    ??? */	xor	%o4,-81,%o4

/*    ??? */	xor	%o5,-97,%o5


/*    104 */	sxar2
/*    104 */	fmuld,s	%f70,%f70,%f74
/*    104 */	fmsubd,sc	%f496,%f162,%f190,%f240

/*    ??? */	sethi	%hi(8240),%o7


/*    104 */	sxar2
/*    ??? */	sethi	%hi(8192),%xg0
/*    104 */	fmsubd,sc	%f128,%f162,%f188,%f224

/*    104 */	sxar1
/*    104 */	fmaddd,s	%f50,%f134,%f40,%f40

/*    ??? */	xor	%o7,-49,%o7


/*    104 */	sxar2
/*    ??? */	xor	%xg0,-1,%xg0
/*    104 */	fmaddd,s	%f186,%f234,%f186,%f186


/*    104 */	sxar2
/*    104 */	fmuld,s	%f238,%f238,%f18
/*    ??? */	sethi	%hi(8320),%xg1


/*    104 */	sxar2
/*    ??? */	sethi	%hi(8304),%xg2
/*    104 */	fmuld,s	%f138,%f138,%f78


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f62,%f144,%f42,%f42
/*    ??? */	xor	%xg1,-129,%xg1


/*    104 */	sxar2
/*    ??? */	xor	%xg2,-113,%xg2
/*    104 */	fmsubd,sc	%f384,%f162,%f188,%f128


/*    104 */	sxar2
/*    104 */	fmuld,s	%f140,%f140,%f82
/*    ??? */	std,s	%f166,[%fp+%o2]


/*    104 */	sxar2
/*    ??? */	sethi	%hi(8240),%xg3
/*    104 */	fnmsubd,s	%f130,%f84,%f0,%f84


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f90,%f162,%f204,%f88
/*    ??? */	xor	%xg3,-49,%xg3


/*    104 */	sxar2
/*    ??? */	sethi	%hi(8320),%xg4
/*    104 */	fnmsubd,s	%f132,%f74,%f0,%f74


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f346,%f162,%f204,%f90
/*    ??? */	xor	%xg4,-129,%xg4


/*    104 */	sxar2
/*    ??? */	sethi	%hi(8224),%xg6
/*    104 */	fmuld,s	%f58,%f224,%f92


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f66,%f142,%f40,%f40
/*    ??? */	xor	%xg6,-33,%xg6


/*    104 */	sxar2
/*    ??? */	sethi	%hi(8240),%xg8
/*    104 */	fmuld,s	%f186,%f186,%f16


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f102,%f162,%f188,%f100
/*    ??? */	xor	%xg8,-49,%xg8


/*    104 */	sxar2
/*    ??? */	sethi	%hi(8304),%xg9
/*    104 */	fmaddd,s	%f134,%f134,%f78,%f78


/*    104 */	sxar2
/*    104 */	fmuld,s	%f42,%f18,%f42
/*    ??? */	xor	%xg9,-113,%xg9


/*    104 */	sxar2
/*    ??? */	sethi	%hi(8208),%xg24
/*    104 */	ldd,s	[%xg23+%xg22],%f106


/*    104 */	sxar2
/*    104 */	fmuld,s	%f54,%f128,%f96
/*    104 */	fmaddd,s	%f136,%f136,%f82,%f82


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg23+%xg11],%f246
/*    104 */	ldd,s	[%xg26+%xg10],%f120


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg26+%xg12],%f122
/*    104 */	fmaddd,s	%f68,%f84,%f68,%f68


/*    104 */	sxar2
/*    104 */	fmuld,s	%f140,%f128,%f30
/*    104 */	ldd,s	[%xg26+%xg13],%f124


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg25+%xg19],%f168
/*    104 */	fmaddd,s	%f70,%f74,%f70,%f70


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f86,%f162,%f198,%f84
/*    104 */	ldd,s	[%xg25+%xg21],%f108


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg25+%xg20],%f104
/*    104 */	fmaddd,s	%f50,%f146,%f92,%f92


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f242,%f162,%f200,%f244
/*    104 */	ldd,s	[%xg25+%xg22],%f172


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg26+%xg15],%f170
/*    104 */	fmuld,s	%f40,%f16,%f40


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f498,%f162,%f200,%f242
/*    104 */	fmsubd,sc	%f120,%f162,%f192,%f116


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f376,%f162,%f192,%f120
/*    ??? */	sethi	%hi(8320),%xg23


/*    104 */	sxar2
/*    ??? */	xor	%xg24,-17,%xg24
/*    104 */	fmsubd,sc	%f122,%f162,%f194,%f118


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f142,%f142,%f78,%f78
/*    ??? */	xor	%xg23,-129,%xg23


/*    104 */	sxar2
/*    ??? */	sethi	%hi(8256),%xg27
/*    104 */	fmaddd,s	%f46,%f148,%f96,%f96


/*    104 */	sxar2
/*    104 */	fmuld,s	%f68,%f68,%f110
/*    ??? */	xor	%xg27,-65,%xg27


/*    104 */	sxar2
/*    ??? */	sethi	%hi(8272),%xg28
/*    ??? */	std,s	%f170,[%fp+%o3]


/*    104 */	sxar2
/*    104 */	fmuld,s	%f70,%f70,%f166
/*    104 */	fmuld,s	%f138,%f224,%f170


/*    104 */	sxar2
/*    ??? */	sethi	%hi(8288),%xg29
/*    104 */	fmsubd,sc	%f378,%f162,%f194,%f122


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f124,%f162,%f196,%f126
/*    ??? */	xor	%xg28,-81,%xg28


/*    104 */	sxar2
/*    ??? */	xor	%xg29,-97,%xg29
/*    104 */	add	%xg5,2,%xg5


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f144,%f144,%f82,%f82
/*    104 */	fmuld,s	%f58,%f244,%f26


/*    104 */	sxar2
/*    ??? */	sethi	%hi(8304),%xg30
/*    104 */	fmaddd,s	%f120,%f120,%f210,%f80


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f380,%f162,%f196,%f124
/*    104 */	sub	%xg7,3,%xg7


/*    104 */	sxar2
/*    ??? */	xor	%xg30,-113,%xg30
/*    104 */	fmuld,s	%f42,%f42,%f236


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f136,%f148,%f30,%f30
/*    104 */	fmaddd,s	%f62,%f240,%f96,%f96


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f130,%f110,%f0,%f110
/*    104 */	ldd,s	[%xg25+%xg11],%f72


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg26+%xg14],%f226
/*    104 */	fnmsubd,s	%f132,%f166,%f0,%f166


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f134,%f146,%f170,%f170
/*    104 */	fmaddd,s	%f116,%f116,%f210,%f76


/*    104 */	sxar2
/*    104 */	fmuld,s	%f40,%f40,%f234
/*    ??? */	sethi	%hi(8304),%xg25


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f342,%f162,%f198,%f86
/*    104 */	fmuld,s	%f54,%f242,%f28


/*    104 */	sxar2
/*    ??? */	xor	%xg25,-113,%xg25
/*    104 */	fmaddd,s	%f122,%f122,%f80,%f80


/*    104 */	sxar2
/*    104 */	fmuld,s	%f112,%f90,%f4
/*    ??? */	std,s	%f226,[%fp+%o4]


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f358,%f162,%f188,%f102
/*    104 */	fmuld,s	%f114,%f100,%f14


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f68,%f110,%f68,%f68
/*    104 */	fmsubd,sc	%f106,%f162,%f206,%f110


/*    104 */	sxar2
/*    104 */	faddd,s	%f82,%f96,%f82
/*    104 */	fmaddd,s	%f70,%f166,%f70,%f70


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f118,%f118,%f76,%f76
/*    104 */	fmsubd,sc	%f98,%f162,%f212,%f96


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f354,%f162,%f212,%f98
/*    104 */	fmsubd,sc	%f362,%f162,%f206,%f106


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f124,%f124,%f80,%f80
/*    104 */	fmaddd,s	%f34,%f86,%f4,%f4


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg26+%xg16],%f74
/*    104 */	ldd,s	[%xg26+%xg17],%f228


/*    104 */	sxar2
/*    104 */	fmuld,s	%f90,%f90,%f250
/*    104 */	fmuld,s	%f112,%f102,%f24


/*    104 */	sxar2
/*    104 */	fmuld,s	%f114,%f88,%f2
/*    104 */	fmuld,s	%f68,%f68,%f226


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f18,%f82,%f236,%f82
/*    104 */	fmuld,s	%f88,%f88,%f252


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f126,%f126,%f76,%f76
/*    104 */	fmaddd,s	%f32,%f96,%f14,%f14


/*    104 */	sxar2
/*    ??? */	std,s	%f228,[%fp+%o5]
/*    104 */	fmuld,s	%f70,%f70,%f228


/*    104 */	sxar2
/*    104 */	frsqrtad,s	%f80,%f248
/*    104 */	fmuld,s	%f80,%f0,%f80


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f86,%f86,%f250,%f250
/*    104 */	fmaddd,s	%f34,%f98,%f24,%f24


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f32,%f84,%f2,%f2
/*    104 */	fnmsubd,s	%f130,%f226,%f0,%f130


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f84,%f84,%f252,%f252
/*    104 */	frsqrtad,s	%f76,%f254


/*    104 */	sxar2
/*    104 */	fmuld,s	%f76,%f0,%f76
/*    ??? */	ldd,s	[%fp+%o7],%f232


/*    104 */	sxar2
/*    ??? */	ldd,s	[%fp+%xg0],%f166
/*    104 */	fnmsubd,s	%f132,%f228,%f0,%f132


/*    104 */	sxar2
/*    ??? */	std,s	%f110,[%fp+%xg1]
/*    104 */	fmsubd,sc	%f168,%f162,%f190,%f110


/*    104 */	sxar2
/*    104 */	fmuld,s	%f248,%f248,%f10
/*    104 */	fmsubd,sc	%f424,%f162,%f190,%f168


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f68,%f130,%f68,%f68
/*    104 */	fmuld,s	%f166,%f40,%f22


/*    104 */	sxar2
/*    104 */	fmuld,s	%f166,%f42,%f20
/*    104 */	fmaddd,s	%f66,%f232,%f92,%f92


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f230,%f162,%f202,%f232
/*    104 */	fmsubd,sc	%f486,%f162,%f202,%f230


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f70,%f132,%f70,%f70
/*    104 */	fmuld,s	%f166,%f82,%f82


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f80,%f10,%f0,%f10
/*    104 */	fmaddd,s	%f38,%f168,%f24,%f24


/*    104 */	sxar2
/*    104 */	fmuld,s	%f68,%f68,%f6
/*    104 */	faddd,s	%f20,%f20,%f226


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f22,%f66,%f142,%f132
/*    104 */	faddd,s	%f78,%f92,%f78


/*    104 */	sxar2
/*    104 */	faddd,s	%f22,%f22,%f228
/*    104 */	fmaddd,s	%f50,%f232,%f26,%f26


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f22,%f50,%f134,%f134
/*    104 */	fnmsubd,s	%f20,%f46,%f136,%f136


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f22,%f58,%f138,%f138
/*    104 */	fmsubd,sc	%f94,%f162,%f208,%f92


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f20,%f54,%f140,%f140
/*    104 */	fmaddd,s	%f46,%f230,%f28,%f28


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f20,%f62,%f144,%f130
/*    104 */	fmsubd,sc	%f350,%f162,%f208,%f94


/*    104 */	sxar2
/*    ??? */	std,s	%f110,[%fp+%xg2]
/*    104 */	fmaddd,s	%f144,%f240,%f30,%f144


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f16,%f78,%f234,%f78
/*    104 */	fmuld,s	%f166,%f22,%f22


/*    104 */	sxar2
/*    104 */	fmuld,s	%f70,%f70,%f8
/*    104 */	fnmsubd,s	%f228,%f134,%f146,%f146


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f226,%f136,%f148,%f148
/*    104 */	fnmsubd,s	%f228,%f138,%f224,%f224


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f226,%f140,%f128,%f128
/*    104 */	fmuld,s	%f166,%f20,%f20


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f62,%f106,%f28,%f28
/*    104 */	fmaddd,s	%f248,%f10,%f248,%f248


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f38,%f94,%f4,%f4
/*    ??? */	ldd,s	[%fp+%xg3],%f12


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f226,%f130,%f240,%f226
/*    ??? */	ldd,s	[%fp+%xg9],%f240


/*    104 */	sxar2
/*    104 */	fmuld,s	%f166,%f78,%f78
/*    104 */	fmuld,s	%f254,%f254,%f110


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f36,%f92,%f2,%f2
/*    104 */	fmaddd,s	%f94,%f94,%f250,%f250


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f92,%f92,%f252,%f252
/*    104 */	fnmsubd,s	%f82,%f46,%f148,%f148


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f142,%f12,%f170,%f142
/*    ??? */	ldd,s	[%fp+%xg4],%f170


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f82,%f54,%f128,%f128
/*    ??? */	ldd,s	[%fp+%xg8],%f12


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f36,%f240,%f14,%f14
/*    104 */	fmaddd,s	%f166,%f144,%f28,%f144


/*    104 */	sxar2
/*    104 */	fmuld,s	%f4,%f8,%f4
/*    104 */	fnmsubd,s	%f82,%f62,%f226,%f226


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f78,%f50,%f146,%f146
/*    104 */	fnmsubd,s	%f78,%f58,%f224,%f224


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f66,%f170,%f26,%f26
/*    ??? */	ldd,s	[%fp+%xg6],%f170


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f76,%f110,%f0,%f110
/*    104 */	fnmsubd,s	%f228,%f132,%f12,%f228


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f20,%f148,%f230,%f230
/*    104 */	faddd,s	%f250,%f24,%f250


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f20,%f128,%f242,%f242
/*    104 */	fmuld,s	%f2,%f6,%f2


/*    104 */	sxar2
/*    104 */	fmuld,s	%f88,%f100,%f240
/*    104 */	fnmsubd,s	%f170,%f234,%f78,%f234


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f170,%f236,%f82,%f236
/*    104 */	fnmsubd,s	%f22,%f146,%f232,%f232


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f22,%f224,%f244,%f244
/*    104 */	fmaddd,s	%f166,%f142,%f26,%f142


/*    104 */	sxar2
/*    104 */	fmuld,s	%f166,%f82,%f82
/*    104 */	fnmsubd,s	%f78,%f66,%f228,%f228


/*    104 */	sxar2
/*    104 */	fmuld,s	%f166,%f78,%f78
/*    104 */	fmuld,s	%f4,%f4,%f12


/*    104 */	sxar2
/*    104 */	fmuld,s	%f248,%f248,%f26
/*    104 */	fmaddd,s	%f254,%f110,%f254,%f254


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f20,%f226,%f106,%f20
/*    104 */	fmuld,s	%f40,%f234,%f40


/*    104 */	sxar2
/*    104 */	fmuld,s	%f42,%f236,%f42
/*    ??? */	ldd,s	[%fp+%xg24],%f234


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f108,%f162,%f200,%f110
/*    104 */	fmaddd,s	%f84,%f96,%f240,%f240


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f82,%f136,%f230,%f230
/*    104 */	fnmsubd,s	%f82,%f140,%f242,%f242


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f78,%f134,%f232,%f232
/*    104 */	fnmsubd,s	%f78,%f138,%f244,%f244


/*    104 */	sxar2
/*    104 */	fmaddd,sc	%f246,%f186,%f234,%f186
/*    104 */	fnmsubd,s	%f80,%f26,%f0,%f26


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f8,%f250,%f12,%f250
/*    104 */	fnmsubd,s	%f82,%f130,%f20,%f82


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f142,%f16,%f40,%f142
/*    104 */	fmaddd,s	%f144,%f18,%f42,%f144


/*    104 */	sxar2
/*    ??? */	ldd,s	[%fp+%xg23],%f40
/*    104 */	fmuld,s	%f254,%f254,%f24


/*    104 */	sxar2
/*    104 */	fmaddd,sc	%f502,%f238,%f234,%f246
/*    104 */	fmuld,s	%f2,%f2,%f10


/*    104 */	sxar2
/*    104 */	faddd,s	%f252,%f14,%f252
/*    104 */	fmsubd,sc	%f104,%f162,%f202,%f106


/*    104 */	sxar2
/*    104 */	fmuld,s	%f114,%f110,%f42
/*    104 */	fnmsubd,s	%f22,%f228,%f40,%f22


/*    104 */	sxar2
/*    104 */	fmuld,s	%f186,%f16,%f186
/*    104 */	fmuld,s	%f166,%f250,%f250


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f248,%f26,%f248,%f248
/*    104 */	fmuld,s	%f166,%f142,%f142


/*    104 */	sxar2
/*    104 */	fmuld,s	%f166,%f144,%f144
/*    104 */	fnmsubd,s	%f76,%f24,%f0,%f24


/*    104 */	sxar2
/*    104 */	fmuld,s	%f246,%f18,%f246
/*    104 */	fmsubd,sc	%f364,%f162,%f200,%f108


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f6,%f252,%f10,%f252
/*    104 */	fmsubd,sc	%f172,%f162,%f206,%f40


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f32,%f106,%f42,%f42
/*    104 */	fnmsubd,s	%f78,%f132,%f22,%f78


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f186,%f138,%f156,%f138
/*    104 */	fmaddd,s	%f186,%f50,%f48,%f48


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f186,%f58,%f56,%f56
/*    104 */	fnmsubd,s	%f142,%f50,%f232,%f232


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f142,%f58,%f244,%f244
/*    104 */	fnmsubd,s	%f144,%f46,%f230,%f230


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f144,%f54,%f242,%f242
/*    104 */	fnmsubd,s	%f144,%f62,%f82,%f144


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f246,%f140,%f154,%f140
/*    ??? */	ldd,s	[%fp+%xg27],%f82


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f254,%f24,%f254,%f254
/*    104 */	fmaddd,s	%f246,%f46,%f44,%f44


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f142,%f66,%f78,%f142
/*    ??? */	ldd,s	[%fp+%xg25],%f78


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f246,%f54,%f52,%f52
/*    104 */	fmaddd,s	%f246,%f62,%f60,%f60


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f186,%f66,%f64,%f64
/*    104 */	fmaddd,s	%f246,%f136,%f150,%f136


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f186,%f134,%f152,%f134
/*    104 */	fmaddd,s	%f246,%f130,%f158,%f130


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f186,%f132,%f160,%f132
/*    104 */	fmaddd,s	%f92,%f78,%f240,%f240


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f246,%f148,%f176,%f148
/*    104 */	fmaddd,s	%f186,%f146,%f174,%f146


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f246,%f128,%f180,%f128
/*    104 */	fmaddd,s	%f186,%f224,%f178,%f224


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f246,%f226,%f184,%f226
/*    104 */	fmaddd,s	%f186,%f228,%f182,%f228


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f246,%f230,%f216,%f230
/*    104 */	fmaddd,s	%f186,%f232,%f214,%f232


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f246,%f242,%f218,%f242
/*    104 */	fmaddd,s	%f186,%f244,%f220,%f244


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f82,%f162,%f204,%f156
/*    104 */	fmsubd,sc	%f338,%f162,%f204,%f154


/*    104 */	sxar2
/*    104 */	fmuld,s	%f254,%f254,%f78
/*    104 */	fmuld,s	%f166,%f252,%f252


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f246,%f144,%f222,%f246
/*    104 */	fmaddd,s	%f186,%f142,%f164,%f186


/*    104 */	sxar2
/*    104 */	fmuld,s	%f248,%f248,%f82
/*    104 */	fmsubd,sc	%f360,%f162,%f202,%f104


/*    104 */	sxar2
/*    104 */	fmuld,s	%f90,%f102,%f236
/*    104 */	fmuld,s	%f112,%f108,%f14


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f36,%f40,%f42,%f42
/*    104 */	ldd,s	[%xg26+%xg18],%f180


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f428,%f162,%f206,%f172
/*    104 */	fmaddd,sc	%f72,%f68,%f234,%f68


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg26+%xg21],%f218
/*    104 */	fmaddd,sc	%f328,%f70,%f234,%f72


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f76,%f78,%f0,%f78
/*    104 */	ldd,s	[%xg26+%xg20],%f216


/*    104 */	sxar2
/*    104 */	fmuld,s	%f166,%f4,%f46
/*    ??? */	ldd,s	[%fp+%xg28],%f142


/*    104 */	sxar2
/*    ??? */	ldd,s	[%fp+%xg29],%f144
/*    104 */	fmuld,s	%f118,%f156,%f54


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f80,%f82,%f0,%f82
/*    104 */	ldd,s	[%xg26+%xg19],%f80


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f180,%f162,%f188,%f178
/*    104 */	fmaddd,s	%f86,%f98,%f236,%f236


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f436,%f162,%f188,%f180
/*    104 */	ldd,s	[%xg26+%xg11],%f222


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f34,%f104,%f14,%f14
/*    104 */	fmaddd,s	%f166,%f240,%f42,%f240


/*    104 */	sxar2
/*    104 */	ldd,s	[%xg26+%xg22],%f164
/*    104 */	fmsubd,sc	%f142,%f162,%f198,%f152


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f144,%f162,%f212,%f174
/*    104 */	fmsubd,sc	%f400,%f162,%f212,%f176


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f254,%f78,%f254,%f254
/*    104 */	fnmsubd,s	%f170,%f10,%f252,%f10


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f170,%f12,%f250,%f12
/*    104 */	fmuld,s	%f166,%f2,%f42


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f398,%f162,%f198,%f150
/*    104 */	fmaddd,s	%f94,%f168,%f236,%f236


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f74,%f162,%f208,%f50
/*    104 */	fmaddd,s	%f38,%f172,%f14,%f14


/*    104 */	sxar2
/*    104 */	fmuld,s	%f122,%f154,%f58
/*    104 */	faddd,s	%f46,%f46,%f66


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f116,%f152,%f54,%f54
/*    104 */	fmuld,s	%f156,%f156,%f70


/*    104 */	sxar2
/*    104 */	fmuld,s	%f154,%f154,%f76
/*    ??? */	ldd,s	[%fp+%xg30],%f158


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f248,%f82,%f248,%f248
/*    104 */	fnmsubd,s	%f46,%f34,%f86,%f86


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f80,%f162,%f190,%f78
/*    104 */	fnmsubd,s	%f46,%f112,%f90,%f90


/*    104 */	sxar2
/*    104 */	fmuld,s	%f118,%f178,%f144
/*    104 */	fmuld,s	%f122,%f180,%f182


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f166,%f236,%f14,%f236
/*    104 */	fmuld,s	%f2,%f10,%f2


/*    104 */	sxar2
/*    104 */	fmuld,s	%f4,%f12,%f4
/*    104 */	fmsubd,sc	%f330,%f162,%f208,%f74


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f120,%f150,%f58,%f58
/*    104 */	fmsubd,sc	%f336,%f162,%f190,%f80


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f126,%f50,%f54,%f54
/*    104 */	fmaddd,s	%f152,%f152,%f70,%f70


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f150,%f150,%f76,%f76
/*    104 */	fmuld,s	%f254,%f254,%f82


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f46,%f38,%f94,%f94
/*    104 */	fnmsubd,s	%f66,%f86,%f98,%f98


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f116,%f174,%f144,%f144
/*    104 */	fmaddd,s	%f120,%f176,%f182,%f182


/*    104 */	sxar2
/*    104 */	fmuld,s	%f72,%f8,%f72
/*    104 */	fnmsubd,s	%f66,%f90,%f102,%f102


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f240,%f6,%f2,%f240
/*    104 */	fmaddd,s	%f236,%f8,%f4,%f236


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f124,%f74,%f58,%f58
/*    104 */	fmuld,s	%f248,%f248,%f142


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f50,%f50,%f70,%f70
/*    104 */	fmaddd,s	%f74,%f74,%f76,%f76


/*    104 */	sxar2
/*    104 */	fmuld,s	%f54,%f82,%f54
/*    104 */	fnmsubd,s	%f66,%f94,%f168,%f66


/*    104 */	sxar2
/*    104 */	fmuld,s	%f156,%f178,%f238
/*    104 */	fmaddd,s	%f126,%f78,%f144,%f144


/*    104 */	sxar2
/*    104 */	faddd,s	%f42,%f42,%f62
/*    104 */	fmaddd,s	%f124,%f80,%f182,%f182


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f42,%f32,%f84,%f84
/*    104 */	fnmsubd,s	%f42,%f114,%f88,%f88


/*    104 */	sxar2
/*    104 */	fmsubd,sc	%f218,%f162,%f200,%f220
/*    104 */	fmuld,s	%f58,%f142,%f58


/*    104 */	sxar2
/*    104 */	fmuld,s	%f68,%f6,%f68
/*    104 */	fmuld,s	%f154,%f180,%f2


/*    104 */	sxar2
/*    104 */	fmuld,s	%f54,%f54,%f168
/*    104 */	fmuld,s	%f166,%f54,%f8


/*    104 */	sxar2
/*    104 */	faddd,s	%f70,%f144,%f70
/*    104 */	fmaddd,s	%f152,%f174,%f238,%f238


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f42,%f36,%f92,%f92
/*    104 */	faddd,s	%f76,%f182,%f76


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f62,%f84,%f96,%f96
/*    104 */	fmsubd,sc	%f474,%f162,%f200,%f218


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f62,%f88,%f100,%f100
/*    104 */	fmsubd,sc	%f216,%f162,%f202,%f214


/*    104 */	sxar2
/*    104 */	fmuld,s	%f58,%f58,%f0
/*    104 */	fmuld,s	%f118,%f220,%f4


/*    104 */	sxar2
/*    104 */	fmuld,s	%f166,%f58,%f10
/*    104 */	fmaddd,s	%f150,%f176,%f2,%f2


/*    104 */	sxar2
/*    104 */	faddd,s	%f8,%f8,%f182
/*    104 */	fnmsubd,s	%f8,%f116,%f152,%f152


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f82,%f70,%f168,%f70
/*    104 */	fnmsubd,s	%f8,%f118,%f156,%f156


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f62,%f92,%f158,%f62
/*    104 */	fmsubd,sc	%f472,%f162,%f202,%f216


/*    104 */	sxar2
/*    104 */	fmuld,s	%f122,%f218,%f6
/*    104 */	fmsubd,sc	%f164,%f162,%f206,%f144


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f116,%f214,%f4,%f4
/*    104 */	fmaddd,s	%f142,%f76,%f0,%f76


/*    104 */	sxar2
/*    104 */	faddd,s	%f10,%f10,%f184
/*    104 */	fnmsubd,s	%f10,%f120,%f150,%f150


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f10,%f122,%f154,%f154
/*    104 */	fnmsubd,s	%f8,%f126,%f50,%f160


/*    104 */	sxar2
/*    104 */	fmuld,s	%f166,%f70,%f70
/*    104 */	fnmsubd,s	%f182,%f152,%f174,%f174


/*    104 */	sxar2
/*    104 */	fmuld,s	%f166,%f42,%f42
/*    104 */	fnmsubd,s	%f182,%f156,%f178,%f178


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f252,%f32,%f96,%f96
/*    104 */	fnmsubd,s	%f252,%f114,%f100,%f100


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f252,%f36,%f62,%f62
/*    104 */	fmsubd,sc	%f420,%f162,%f206,%f164


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f120,%f216,%f6,%f6
/*    104 */	fnmsubd,s	%f10,%f124,%f74,%f158


/*    104 */	sxar2
/*    104 */	fmuld,s	%f166,%f76,%f76
/*    104 */	fmaddd,s	%f50,%f78,%f238,%f50


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f126,%f144,%f4,%f4
/*    104 */	fnmsubd,s	%f184,%f150,%f176,%f176


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f170,%f168,%f70,%f168
/*    104 */	fnmsubd,s	%f184,%f154,%f180,%f180


/*    104 */	sxar2
/*    104 */	fmuld,s	%f166,%f46,%f46
/*    104 */	fnmsubd,s	%f182,%f160,%f78,%f182


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f250,%f34,%f98,%f98
/*    104 */	fnmsubd,s	%f250,%f112,%f102,%f102


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f250,%f38,%f66,%f66
/*    104 */	fmuld,s	%f166,%f252,%f252


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f42,%f96,%f106,%f106
/*    104 */	fnmsubd,s	%f42,%f100,%f110,%f110


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f74,%f80,%f2,%f74
/*    104 */	fmaddd,s	%f124,%f164,%f6,%f6


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f170,%f0,%f76,%f0
/*    104 */	fnmsubd,s	%f184,%f158,%f80,%f184


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f166,%f50,%f4,%f50
/*    104 */	fmuld,s	%f54,%f168,%f54


/*    104 */	sxar2
/*    104 */	fmuld,s	%f166,%f8,%f8
/*    104 */	fnmsubd,s	%f70,%f116,%f174,%f174


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f70,%f118,%f178,%f178
/*    104 */	fnmsubd,s	%f70,%f126,%f182,%f182


/*    104 */	sxar2
/*    104 */	fmuld,s	%f166,%f250,%f250
/*    104 */	fnmsubd,s	%f46,%f98,%f104,%f104


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f46,%f102,%f108,%f108
/*    104 */	fnmsubd,s	%f42,%f62,%f40,%f42


/*    104 */	sxar2
/*    104 */	fmuld,s	%f166,%f240,%f240
/*    104 */	fnmsubd,s	%f252,%f84,%f106,%f106


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f252,%f88,%f110,%f110
/*    104 */	fmaddd,s	%f166,%f74,%f6,%f74


/*    104 */	sxar2
/*    104 */	fmuld,s	%f58,%f0,%f58
/*    104 */	fmuld,s	%f166,%f10,%f10


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f76,%f120,%f176,%f176
/*    104 */	fnmsubd,s	%f76,%f122,%f180,%f180


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f76,%f124,%f184,%f184
/*    104 */	fmaddd,s	%f50,%f82,%f54,%f50


/*    104 */	sxar2
/*    104 */	fmuld,s	%f166,%f70,%f70
/*    104 */	fnmsubd,s	%f8,%f174,%f214,%f214


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f46,%f66,%f172,%f46
/*    104 */	fnmsubd,s	%f8,%f178,%f220,%f220


/*    104 */	sxar2
/*    104 */	fmuld,s	%f166,%f236,%f236
/*    104 */	fnmsubd,s	%f250,%f86,%f104,%f104


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f250,%f90,%f108,%f108
/*    104 */	fnmsubd,s	%f252,%f92,%f42,%f252


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f240,%f32,%f106,%f106
/*    104 */	fnmsubd,s	%f240,%f114,%f110,%f110


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f74,%f142,%f58,%f74
/*    104 */	fmuld,s	%f166,%f76,%f76


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f10,%f176,%f216,%f216
/*    104 */	fnmsubd,s	%f10,%f180,%f218,%f218


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f8,%f182,%f144,%f8
/*    104 */	fmaddd,sc	%f222,%f254,%f234,%f254


/*    104 */	sxar2
/*    104 */	fmuld,s	%f166,%f50,%f50
/*    104 */	fnmsubd,s	%f70,%f152,%f214,%f214


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f250,%f94,%f46,%f250
/*    104 */	fnmsubd,s	%f70,%f156,%f220,%f220


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f236,%f34,%f104,%f104
/*    104 */	fnmsubd,s	%f236,%f112,%f108,%f108


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f240,%f36,%f252,%f240
/*    104 */	fnmsubd,s	%f10,%f184,%f164,%f10


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f68,%f32,%f48,%f48
/*    104 */	fmaddd,s	%f68,%f114,%f56,%f56


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f68,%f36,%f64,%f64
/*    104 */	fmaddd,s	%f68,%f84,%f134,%f84


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f68,%f88,%f138,%f88
/*    104 */	fmaddd,s	%f68,%f92,%f132,%f92


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f68,%f96,%f146,%f96
/*    104 */	fmaddd,s	%f68,%f100,%f224,%f100


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f68,%f62,%f228,%f62
/*    104 */	fmaddd,s	%f68,%f106,%f232,%f106


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f68,%f110,%f244,%f110
/*    104 */	fmaddd,sc	%f478,%f248,%f234,%f222


/*    104 */	sxar2
/*    104 */	fmuld,s	%f166,%f74,%f74
/*    104 */	fnmsubd,s	%f76,%f150,%f216,%f216


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f76,%f154,%f218,%f218
/*    104 */	fnmsubd,s	%f70,%f160,%f8,%f70


/*    104 */	sxar2
/*    104 */	fmuld,s	%f254,%f82,%f254
/*    104 */	fnmsubd,s	%f50,%f116,%f214,%f214


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f236,%f38,%f250,%f236
/*    104 */	fnmsubd,s	%f50,%f118,%f220,%f220


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f72,%f34,%f44,%f44
/*    104 */	fmaddd,s	%f72,%f112,%f52,%f52


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f72,%f38,%f60,%f60
/*    104 */	fmaddd,s	%f72,%f86,%f136,%f86


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f72,%f90,%f140,%f90
/*    104 */	fmaddd,s	%f72,%f94,%f130,%f94


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f72,%f98,%f148,%f98
/*    104 */	fmaddd,s	%f72,%f102,%f128,%f102


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f72,%f66,%f226,%f66
/*    104 */	fmaddd,s	%f72,%f104,%f230,%f104


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f72,%f108,%f242,%f108
/*    104 */	fmaddd,s	%f68,%f240,%f186,%f68


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f76,%f158,%f10,%f76
/*    104 */	fmuld,s	%f222,%f142,%f222


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f74,%f120,%f216,%f216
/*    104 */	fnmsubd,s	%f74,%f122,%f218,%f218


/*    104 */	sxar2
/*    104 */	fnmsubd,s	%f50,%f126,%f70,%f50
/*    104 */	fmaddd,s	%f254,%f116,%f48,%f48


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f254,%f118,%f56,%f56
/*    104 */	fmaddd,s	%f254,%f126,%f64,%f64


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f254,%f152,%f84,%f152
/*    104 */	fmaddd,s	%f254,%f156,%f88,%f156


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f254,%f160,%f92,%f160
/*    104 */	fmaddd,s	%f254,%f174,%f96,%f174


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f254,%f178,%f100,%f178
/*    104 */	fmaddd,s	%f254,%f182,%f62,%f182


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f254,%f214,%f106,%f214
/*    104 */	fmaddd,s	%f254,%f220,%f110,%f220


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f72,%f236,%f246,%f72
/*    104 */	fnmsubd,s	%f74,%f124,%f76,%f74


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f222,%f120,%f44,%f44
/*    104 */	fmaddd,s	%f222,%f122,%f52,%f52


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f222,%f124,%f60,%f60
/*    104 */	fmaddd,s	%f222,%f150,%f86,%f150


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f222,%f154,%f90,%f154
/*    104 */	fmaddd,s	%f222,%f158,%f94,%f158


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f222,%f176,%f98,%f176
/*    104 */	fmaddd,s	%f222,%f180,%f102,%f180


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f222,%f184,%f66,%f184
/*    104 */	fmaddd,s	%f222,%f216,%f104,%f216


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f222,%f218,%f108,%f218
/*    104 */	fmaddd,s	%f254,%f50,%f68,%f254


/*    104 */	sxar2
/*    104 */	fmaddd,s	%f222,%f74,%f72,%f222
/*    104 */	fmovd,s	%f254,%f164

.L7085:


.L7084:


.L7087:

/*    105 */	sxar1
/*    105 */	srl	%xg5,31,%xg30

/* #00002 */	sethi	%hi(8160),%g1

/*    105 */	sxar1
/*    105 */	add	%xg30,%xg5,%xg30

/* #00002 */	xor	%g1,-993,%g1


/*    136 */	sxar2
/*    136 */	sra	%xg30,1,%xg30
/* #00002 */	ldd,s	[%fp+%g1],%f126

/* #00002 */	sethi	%hi(8176),%g2

/*    105 */	sxar1
/*    105 */	sra	%xg30,%g0,%xg30

/* #00002 */	xor	%g2,-1009,%g2


/*     38 */	sxar2
/*     38 */	mulx	%xg30,208,%xg30
/* #00002 */	ldd,s	[%fp+%g2],%f128

/* #00002 */	sethi	%hi(8192),%g3

/* #00002 */	xor	%g3,-1,%g3

/* #00002 */	sethi	%hi(8208),%g4

/*     44 */	sxar1
/* #00002 */	ldd,s	[%fp+%g3],%f142

/* #00002 */	xor	%g4,-17,%g4

/* #00002 */	sethi	%hi(8224),%g5

/*    153 */	sxar1
/* #00002 */	ldd,s	[%fp+%g4],%f144

/* #00002 */	xor	%g5,-33,%g5


/*     54 */	sxar2
/*     54 */	add	%xg5,2,%xg5
/* #00002 */	ldd,s	[%fp+%g5],%f146


/*     19 */	sxar2
/*     19 */	subcc	%xg7,1,%xg7
/*     19 */	ldd,s	[%xg10+%xg30],%f130


/*    136 */	sxar2
/*    136 */	ldd,s	[%xg12+%xg30],%f134
/*    136 */	ldd,s	[%xg13+%xg30],%f230


/*    136 */	sxar2
/*    136 */	ldd,s	[%xg16+%xg30],%f242
/*    136 */	ldd,s	[%xg14+%xg30],%f234


/*    136 */	sxar2
/*    136 */	ldd,s	[%xg17+%xg30],%f246
/*    136 */	ldd,s	[%xg15+%xg30],%f238


/*    177 */	sxar2
/*    177 */	fmsubd,sc	%f130,%f126,%f192,%f132
/*    177 */	fmsubd,sc	%f386,%f126,%f192,%f130


/*    177 */	sxar2
/*    177 */	fmsubd,sc	%f134,%f126,%f194,%f136
/*    177 */	fmsubd,sc	%f390,%f126,%f194,%f134


/*    177 */	sxar2
/*    177 */	fmsubd,sc	%f230,%f126,%f196,%f228
/*    177 */	fmsubd,sc	%f486,%f126,%f196,%f230


/*    177 */	sxar2
/*    177 */	fmsubd,sc	%f242,%f126,%f208,%f240
/*    177 */	fmsubd,sc	%f498,%f126,%f208,%f242


/*    136 */	sxar2
/*    136 */	fmsubd,sc	%f234,%f126,%f198,%f232
/*    136 */	fmsubd,sc	%f238,%f126,%f204,%f236


/*    136 */	sxar2
/*    136 */	fmsubd,sc	%f494,%f126,%f204,%f238
/*    136 */	fmsubd,sc	%f246,%f126,%f212,%f244


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f132,%f132,%f210,%f36
/*     44 */	fmaddd,s	%f130,%f130,%f210,%f38


/*    177 */	sxar2
/*    177 */	fmsubd,sc	%f490,%f126,%f198,%f234
/*    177 */	fmsubd,sc	%f502,%f126,%f212,%f246


/*    136 */	sxar2
/*    136 */	ldd,s	[%xg18+%xg30],%f250
/*    136 */	ldd,s	[%xg19+%xg30],%f254


/*     44 */	sxar2
/*     44 */	fmuld,s	%f136,%f236,%f54
/*     44 */	fmuld,s	%f236,%f236,%f62


/*     44 */	sxar2
/*     44 */	fmuld,s	%f134,%f238,%f58
/*     44 */	fmuld,s	%f238,%f238,%f66


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f136,%f136,%f36,%f36
/*     44 */	fmaddd,s	%f134,%f134,%f38,%f38


/*    136 */	sxar2
/*    136 */	ldd,s	[%xg20+%xg30],%f34
/*    136 */	fmsubd,sc	%f250,%f126,%f188,%f248


/*    136 */	sxar2
/*    136 */	fmsubd,sc	%f506,%f126,%f188,%f250
/*    136 */	fmsubd,sc	%f254,%f126,%f190,%f252


/*     44 */	sxar2
/*     44 */	fmsubd,sc	%f510,%f126,%f190,%f254
/*     44 */	fmaddd,s	%f132,%f232,%f54,%f54


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f232,%f232,%f62,%f62
/*     44 */	fmaddd,s	%f130,%f234,%f58,%f58


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f234,%f234,%f66,%f66
/*     44 */	fmaddd,s	%f228,%f228,%f36,%f36


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f230,%f230,%f38,%f38
/*     44 */	fmuld,s	%f136,%f248,%f68


/*    136 */	sxar2
/*    136 */	fmuld,s	%f134,%f250,%f70
/*    136 */	ldd,s	[%xg21+%xg30],%f42


/*     44 */	sxar2
/*     44 */	ldd,s	[%xg22+%xg30],%f50
/*     44 */	fmuld,s	%f236,%f248,%f72


/*    136 */	sxar2
/*    136 */	fmuld,s	%f238,%f250,%f74
/*    136 */	fmsubd,sc	%f34,%f126,%f202,%f32


/*    153 */	sxar2
/*    153 */	fmsubd,sc	%f290,%f126,%f202,%f34
/*    153 */	ldd,s	[%xg11+%xg30],%f138


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f228,%f240,%f54,%f54
/*     44 */	fmaddd,s	%f240,%f240,%f62,%f62


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f230,%f242,%f58,%f58
/*     44 */	fmaddd,s	%f242,%f242,%f66,%f66


/*     60 */	sxar2
/*     60 */	frsqrtad,s	%f36,%f140
/*     60 */	frsqrtad,s	%f38,%f84


/*     38 */	sxar2
/*     38 */	fmuld,s	%f36,%f128,%f36
/*     38 */	fmuld,s	%f38,%f128,%f38


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f132,%f244,%f68,%f68
/*     44 */	fmaddd,s	%f130,%f246,%f70,%f70


/*    177 */	sxar2
/*    177 */	fmsubd,sc	%f42,%f126,%f200,%f40
/*    177 */	fmsubd,sc	%f298,%f126,%f200,%f42


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f232,%f244,%f72,%f72
/*     44 */	fmaddd,s	%f234,%f246,%f74,%f74


/*    177 */	sxar2
/*    177 */	fmsubd,sc	%f50,%f126,%f206,%f46
/*    177 */	fmsubd,sc	%f306,%f126,%f206,%f50


/*     32 */	sxar2
/*     32 */	fmuld,s	%f140,%f140,%f80
/*     32 */	fmuld,s	%f84,%f84,%f88


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f228,%f252,%f68,%f68
/*     44 */	fmaddd,s	%f230,%f254,%f70,%f70


/*     44 */	sxar2
/*     44 */	fmuld,s	%f136,%f40,%f76
/*     44 */	fmuld,s	%f134,%f42,%f78


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f240,%f252,%f72,%f72
/*     44 */	fmaddd,s	%f242,%f254,%f74,%f74


/*     32 */	sxar2
/*     32 */	fnmsubd,s	%f36,%f80,%f128,%f80
/*     32 */	fnmsubd,s	%f38,%f88,%f128,%f88


/*     44 */	sxar2
/*     44 */	faddd,s	%f62,%f68,%f62
/*     44 */	faddd,s	%f66,%f70,%f66


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f132,%f32,%f76,%f76
/*     44 */	fmaddd,s	%f130,%f34,%f78,%f78


/*     32 */	sxar2
/*     32 */	fmaddd,s	%f140,%f80,%f140,%f140
/*     32 */	fmaddd,s	%f84,%f88,%f84,%f84


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f228,%f46,%f76,%f76
/*     44 */	fmaddd,s	%f230,%f50,%f78,%f78


/*     32 */	sxar2
/*     32 */	fmuld,s	%f140,%f140,%f82
/*     32 */	fmuld,s	%f84,%f84,%f90


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f142,%f72,%f76,%f72
/*     44 */	fmaddd,s	%f142,%f74,%f78,%f74


/*     32 */	sxar2
/*     32 */	fnmsubd,s	%f36,%f82,%f128,%f82
/*     32 */	fnmsubd,s	%f38,%f90,%f128,%f90


/*     32 */	sxar2
/*     32 */	fmaddd,s	%f140,%f82,%f140,%f140
/*     32 */	fmaddd,s	%f84,%f90,%f84,%f84


/*     32 */	sxar2
/*     32 */	fmuld,s	%f140,%f140,%f86
/*     32 */	fmuld,s	%f84,%f84,%f92


/*     32 */	sxar2
/*     32 */	fnmsubd,s	%f36,%f86,%f128,%f36
/*     32 */	fnmsubd,s	%f38,%f92,%f128,%f38


/*     32 */	sxar2
/*     32 */	fmaddd,s	%f140,%f36,%f140,%f140
/*     32 */	fmaddd,s	%f84,%f38,%f84,%f84


/*     54 */	sxar2
/*     54 */	fmuld,s	%f140,%f140,%f94
/*     54 */	fmuld,s	%f84,%f84,%f98


/*    194 */	sxar2
/*    194 */	fmaddd,sc	%f138,%f140,%f144,%f140
/*    194 */	fmaddd,sc	%f394,%f84,%f144,%f138


/*     54 */	sxar2
/*     54 */	fmuld,s	%f54,%f94,%f54
/*     54 */	fmuld,s	%f58,%f98,%f58


/*     54 */	sxar2
/*     54 */	fmuld,s	%f140,%f94,%f140
/*     54 */	fmuld,s	%f138,%f98,%f138


/*     54 */	sxar2
/*     54 */	fmuld,s	%f54,%f54,%f96
/*     54 */	fmuld,s	%f58,%f58,%f100


/*     54 */	sxar2
/*     54 */	fmuld,s	%f142,%f54,%f102
/*     54 */	fmuld,s	%f142,%f58,%f104


/*     24 */	sxar2
/*     24 */	fmaddd,s	%f138,%f130,%f44,%f44
/*     24 */	fmaddd,s	%f140,%f132,%f48,%f48


/*     24 */	sxar2
/*     24 */	fmaddd,s	%f138,%f134,%f52,%f52
/*     24 */	fmaddd,s	%f140,%f136,%f56,%f56


/*     24 */	sxar2
/*     24 */	fmaddd,s	%f138,%f230,%f60,%f60
/*     24 */	fmaddd,s	%f140,%f228,%f64,%f64


/*     44 */	sxar2
/*     44 */	fmaddd,s	%f94,%f62,%f96,%f62
/*     44 */	fmaddd,s	%f98,%f66,%f100,%f66


/*     44 */	sxar2
/*     44 */	faddd,s	%f102,%f102,%f106
/*     44 */	faddd,s	%f104,%f104,%f108


/*     49 */	sxar2
/*     49 */	fnmsubd,s	%f104,%f130,%f234,%f234
/*     49 */	fnmsubd,s	%f102,%f132,%f232,%f232


/*     49 */	sxar2
/*     49 */	fnmsubd,s	%f102,%f136,%f236,%f236
/*     49 */	fnmsubd,s	%f104,%f134,%f238,%f238


/*     54 */	sxar2
/*     54 */	fmuld,s	%f142,%f102,%f110
/*     54 */	fmuld,s	%f142,%f104,%f112


/*     49 */	sxar2
/*     49 */	fnmsubd,s	%f102,%f228,%f240,%f102
/*     49 */	fnmsubd,s	%f104,%f230,%f242,%f104


/*     54 */	sxar2
/*     54 */	fmuld,s	%f142,%f62,%f62
/*     54 */	fmuld,s	%f142,%f66,%f66


/*     49 */	sxar2
/*     49 */	fnmsubd,s	%f106,%f232,%f244,%f244
/*     49 */	fnmsubd,s	%f108,%f234,%f246,%f246


/*     49 */	sxar2
/*     49 */	fnmsubd,s	%f106,%f236,%f248,%f248
/*     49 */	fnmsubd,s	%f108,%f238,%f250,%f250


/*     24 */	sxar2
/*     24 */	fmaddd,s	%f138,%f234,%f150,%f150
/*     24 */	fmaddd,s	%f140,%f232,%f152,%f152


/*     49 */	sxar2
/*     49 */	fnmsubd,s	%f106,%f102,%f252,%f106
/*     49 */	fnmsubd,s	%f108,%f104,%f254,%f108


/*     54 */	sxar2
/*     54 */	fnmsubd,s	%f146,%f96,%f62,%f96
/*     54 */	fnmsubd,s	%f146,%f100,%f66,%f100


/*     54 */	sxar2
/*     54 */	fmuld,s	%f142,%f62,%f114
/*     54 */	fmuld,s	%f142,%f66,%f116


/*     49 */	sxar2
/*     49 */	fnmsubd,s	%f62,%f132,%f244,%f244
/*     49 */	fnmsubd,s	%f66,%f130,%f246,%f246


/*     49 */	sxar2
/*     49 */	fnmsubd,s	%f62,%f136,%f248,%f248
/*     49 */	fnmsubd,s	%f66,%f134,%f250,%f250


/*     24 */	sxar2
/*     24 */	fmaddd,s	%f138,%f238,%f154,%f154
/*     24 */	fmaddd,s	%f140,%f236,%f156,%f156


/*     49 */	sxar2
/*     49 */	fnmsubd,s	%f62,%f228,%f106,%f62
/*     49 */	fnmsubd,s	%f66,%f230,%f108,%f66


/*     54 */	sxar2
/*     54 */	fmuld,s	%f54,%f96,%f54
/*     54 */	fmuld,s	%f58,%f100,%f58


/*     24 */	sxar2
/*     24 */	fmaddd,s	%f138,%f104,%f158,%f158
/*     24 */	fmaddd,s	%f140,%f102,%f160,%f160


/*     24 */	sxar2
/*     24 */	fmaddd,s	%f138,%f246,%f176,%f176
/*     24 */	fmaddd,s	%f140,%f244,%f174,%f174


/*     24 */	sxar2
/*     24 */	fmaddd,s	%f138,%f250,%f180,%f180
/*     24 */	fmaddd,s	%f140,%f248,%f178,%f178


/*     24 */	sxar2
/*     24 */	fnmsubd,s	%f112,%f246,%f34,%f246
/*     24 */	fnmsubd,s	%f110,%f244,%f32,%f244


/*     24 */	sxar2
/*     24 */	fnmsubd,s	%f112,%f250,%f42,%f250
/*     24 */	fnmsubd,s	%f110,%f248,%f40,%f248


/*     54 */	sxar2
/*     54 */	fmaddd,s	%f72,%f94,%f54,%f72
/*     54 */	fmaddd,s	%f74,%f98,%f58,%f74


/*     24 */	sxar2
/*     24 */	fnmsubd,s	%f112,%f66,%f50,%f112
/*     24 */	fnmsubd,s	%f110,%f62,%f46,%f110


/*     24 */	sxar2
/*     24 */	fmaddd,s	%f138,%f66,%f184,%f184
/*     24 */	fmaddd,s	%f140,%f62,%f182,%f182


/*     24 */	sxar2
/*     24 */	fnmsubd,s	%f116,%f234,%f246,%f234
/*     24 */	fnmsubd,s	%f114,%f232,%f244,%f232


/*     24 */	sxar2
/*     24 */	fnmsubd,s	%f116,%f238,%f250,%f238
/*     24 */	fnmsubd,s	%f114,%f236,%f248,%f236


/*     54 */	sxar2
/*     54 */	fmuld,s	%f142,%f72,%f72
/*     54 */	fmuld,s	%f142,%f74,%f74


/*     24 */	sxar2
/*     24 */	fnmsubd,s	%f116,%f104,%f112,%f116
/*     24 */	fnmsubd,s	%f114,%f102,%f110,%f114


/*     24 */	sxar2
/*     24 */	fnmsubd,s	%f74,%f130,%f234,%f130
/*     24 */	fnmsubd,s	%f72,%f132,%f232,%f132


/*     24 */	sxar2
/*     24 */	fnmsubd,s	%f74,%f134,%f238,%f134
/*     24 */	fnmsubd,s	%f72,%f136,%f236,%f136


/*     24 */	sxar2
/*     24 */	fnmsubd,s	%f74,%f230,%f116,%f74
/*     24 */	fnmsubd,s	%f72,%f228,%f114,%f72


/*     24 */	sxar2
/*     24 */	fmaddd,s	%f138,%f130,%f216,%f130
/*     24 */	fmaddd,s	%f140,%f132,%f214,%f132


/*     24 */	sxar2
/*     24 */	fmaddd,s	%f138,%f134,%f218,%f134
/*     24 */	fmaddd,s	%f140,%f136,%f220,%f136


/*     24 */	sxar2
/*     24 */	fmaddd,s	%f138,%f74,%f222,%f138
/*     24 */	fmaddd,s	%f140,%f72,%f164,%f140


/*     24 */	sxar2
/*     24 */	fmovd,s	%f130,%f216
/*     24 */	fmovd,s	%f132,%f214


/*     24 */	sxar2
/*     24 */	fmovd,s	%f134,%f218
/*     24 */	fmovd,s	%f136,%f220


/*     24 */	sxar2
/*     24 */	fmovd,s	%f138,%f222
/*     24 */	fmovd,s	%f140,%f164

/*    195 */	bne,pt	%icc, .L7087
	nop


.L7083:


/*    195 */	sxar2
/*    195 */	std,s	%f48,[%i2+-4032]
/*    195 */	std,s	%f44,[%i2+-4016]


/*    195 */	sxar2
/*    195 */	std,s	%f56,[%i2+-4000]
/*    195 */	std,s	%f52,[%i2+-3984]


/*    195 */	sxar2
/*    195 */	std,s	%f64,[%i2+-3968]
/*    195 */	std,s	%f60,[%i2+-3952]


/*    195 */	sxar2
/*    195 */	std,s	%f152,[%i2+-3936]
/*    195 */	std,s	%f150,[%i2+-3920]


/*    195 */	sxar2
/*    195 */	std,s	%f156,[%i2+-3904]
/*    195 */	std,s	%f154,[%i2+-3888]


/*    195 */	sxar2
/*    195 */	std,s	%f160,[%i2+-3872]
/*    195 */	std,s	%f158,[%i2+-3856]


/*    195 */	sxar2
/*    195 */	std,s	%f174,[%i2+-3840]
/*    195 */	std,s	%f176,[%i2+-3824]


/*    195 */	sxar2
/*    195 */	std,s	%f178,[%i2+-3808]
/*    195 */	std,s	%f180,[%i2+-3792]


/*    195 */	sxar2
/*    195 */	std,s	%f182,[%i2+-3776]
/*    195 */	std,s	%f184,[%i2+-3760]


/*    195 */	sxar2
/*    195 */	std,s	%f214,[%i2+-3744]
/*    195 */	std,s	%f216,[%i2+-3728]


/*    195 */	sxar2
/*    195 */	std,s	%f220,[%i2+-3712]
/*    195 */	std,s	%f218,[%i2+-3696]


/*    195 */	sxar2
/*    195 */	std,s	%f164,[%i2+-3680]
/*    195 */	std,s	%f222,[%i2+-3664]

.L6939:

/*    178 *//*    178 */	call	__mpc_obar
/*    178 */	ldx	[%fp+2199],%o0

/*    178 */

/*     88 */	sxar2
/*     88 */	add	%l7,%l4,%xg27
/*     88 */	ldd,s	[%i2+-4032],%f148
/*     88 */	sxar1
/*     88 */	ldd,s	[%i2+-4016],%f146

/*    184 */	add	%l0,2,%l0

/*     88 */	sxar2
/*     88 */	ldd,s	[%i2+-4000],%f152
/*     88 */	ldd,s	[%i2+-3984],%f150


/*     88 */	sxar2
/*     88 */	ldd,s	[%i2+-3968],%f156
/*     88 */	ldd,s	[%i2+-3952],%f154
/*     88 */	sxar2
/*     88 */	ldd,s	[%i2+-3936],%f160
/*     88 */	ldd,s	[%i2+-3920],%f158
/*     88 */	sxar2
/*     88 */	faddd,s	%f148,%f146,%f148
/*     88 */	ldd,s	[%i2+-3904],%f164
/*     88 */	sxar2
/*     88 */	ldd,s	[%i2+-3888],%f162
/*     88 */	faddd,s	%f152,%f150,%f152
/*     88 */	sxar2
/*     88 */	ldd,s	[%i2+-3872],%f168
/*     88 */	ldd,s	[%i2+-3856],%f166
/*     88 */	sxar2
/*     88 */	faddd,s	%f156,%f154,%f156
/*     88 */	ldd,s	[%i2+-3840],%f172
/*     88 */	sxar2
/*     88 */	ldd,s	[%i2+-3824],%f170
/*     88 */	faddd,s	%f160,%f158,%f160
/*     88 */	sxar2
/*     88 */	ldd,s	[%i2+-3808],%f176
/*     88 */	ldd,s	[%i2+-3792],%f174
/*     88 */	sxar2
/*     88 */	faddd,s	%f164,%f162,%f164
/*     88 */	ldd,s	[%i2+-3776],%f180
/*     88 */	sxar2
/*     88 */	ldd,s	[%i2+-3760],%f178
/*     88 */	faddd,s	%f168,%f166,%f168
/*     88 */	sxar2
/*     88 */	ldd,s	[%i2+-3744],%f184
/*     88 */	ldd,s	[%i2+-3728],%f182
/*     88 */	sxar2
/*     88 */	faddd,s	%f172,%f170,%f172
/*     88 */	ldd,s	[%i2+-3712],%f188
/*     88 */	sxar2
/*     88 */	ldd,s	[%i2+-3696],%f186
/*     88 */	faddd,s	%f176,%f174,%f176
/*     88 */	sxar2
/*     88 */	ldd,s	[%i2+-3680],%f192
/*     88 */	ldd,s	[%i2+-3664],%f190
/*     88 */	sxar2
/*     88 */	faddd,s	%f180,%f178,%f180
/*     88 */	faddd,s	%f184,%f182,%f184
/*     88 */	sxar2
/*     88 */	faddd,s	%f188,%f186,%f188
/*     88 */	faddd,s	%f192,%f190,%f192

/*     21 */	sxar2
/*     21 */	std,s	%f148,[%xg27]
/*     21 */	std,s	%f152,[%xg27+16]
/*     21 */	sxar2
/*     21 */	std,s	%f156,[%xg27+32]
/*     21 */	std,s	%f160,[%xg27+48]
/*     21 */	sxar2
/*     21 */	std,s	%f164,[%xg27+64]
/*     21 */	std,s	%f168,[%xg27+80]
/*     21 */	sxar2
/*     21 */	std,s	%f172,[%xg27+96]
/*     21 */	std,s	%f176,[%xg27+112]
/*     21 */	sxar2
/*     21 */	std,s	%f180,[%xg27+128]
/*     21 */	std,s	%f184,[%xg27+144]
/*     21 */	sxar2
/*     21 */	std,s	%f188,[%xg27+160]
/*     21 */	std,s	%f192,[%xg27+176]

/*    184 */	sxar2
/*    184 */	ldsw	[%i0+2195],%xg31
/*    184 */	cmp	%l0,%xg31
/*    184 */	bl,pt	%icc, .L6926
/*    184 */	add	%l4,192,%l4


.L6940:


.L6941:


/*    185 */	call	__mpc_obar
/*    185 */	ldx	[%fp+2199],%o0


.L6942:

/*    185 */	ret
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
	.uleb128	.L6920-.LLFB7
	.uleb128	0x0
.LLLSDACSE7:
	.sleb128	0
	.sleb128	0
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite8-k.cpp _ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_3 $"
	.section	".text"
	.align	64
_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_3:
.LLFB8:
.L6944:

/*    187 */	save	%sp,-496,%sp
.LLCFI6:
/*    187 */	stx	%i0,[%fp+2175]
/*    187 */	stx	%i3,[%fp+2199]
/*    187 */	stx	%i0,[%fp+2175]

.L6945:

/*    187 *//*    187 */	sxar1
/*    187 */	ldsw	[%i0+2035],%xg18
/*    187 */
/*    187 */
/*    187 */
/*    188 */	ldsw	[%i0+2187],%o0
/*    188 */	ldsw	[%i0+2195],%l0
/*    188 */	cmp	%o0,%l0
/*    188 */	bge	.L6958
	nop


.L6946:

/*    188 */	sxar1
/*    188 */	mov	1,%xg17

/*    188 */	sra	%l0,%g0,%l0


/*    188 */	sxar2
/*    188 */	fzero,s	%f108
/*    188 */	stx	%xg17,[%fp+2031]

/*    188 */	sxar1
/*    ??? */	std,s	%f108,[%fp+1775]

.L6947:

/*    188 */	sethi	%h44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%l1

/*    188 */	mov	1,%l6

/*    188 */	or	%l1,%m44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%l1

/*    188 */	sethi	%hi(102336),%l7

/*    188 */	sllx	%l1,12,%l1

/*    188 */	add	%fp,2039,%l2

/*    188 */	or	%l1,%l44(_ZZN7Gravity19calc_force_in_rangeEiidP5ForceE5fobuf),%l1

/*    188 */	add	%fp,2023,%l3

/*    188 */	add	%fp,2031,%l4

/*    188 */	sra	%l6,%g0,%l5

/*    188 */	or	%l7,960,%l7

/*    188 */	sethi	%hi(196608),%i2

/*    188 */	sethi	%hi(98304),%i1

.L6948:

/*    188 */	sra	%o0,%g0,%o0

/*    188 */	stx	%g0,[%sp+2223]

/*    188 */	mov	2,%o2

/*    188 */	mov	%g0,%o3

/*    188 */	mov	%l0,%o1

/*    188 */	mov	%l2,%o4


/*    188 */	stx	%g0,[%sp+2231]

/*    188 */	stx	%l4,[%sp+2239]


/*    188 */	sxar2
/*    188 */	ldx	[%fp+2199],%xg15
/*    188 */	stx	%xg15,[%sp+2247]

/*    188 */	call	__mpc_ostd_th
/*    188 */	mov	%l3,%o5
/*    188 */	sxar2
/*    188 */	ldx	[%fp+2031],%xg16
/*    188 */	cmp	%xg16,%g0
/*    188 */	ble,pn	%xcc, .L6958
	nop


.L6949:

/*    188 */	ldx	[%fp+2039],%o0


/*    188 */	sxar2
/*    188 */	ldx	[%fp+2023],%xg0
/*    188 */	ldsw	[%i0+2187],%xg7


/*    188 */	sxar2
/*    188 */	ldx	[%i0+2207],%xg8
/*    188 */	ldsw	[%i0+2027],%xg11

/*    188 */	sra	%o0,%g0,%o0


/*    188 */	sxar2
/*    188 */	sra	%xg0,%g0,%xg0
/*    188 */	sub	%xg0,%o0,%xg0


/*    188 */	sxar2
/*    188 */	add	%o0,1,%xg1
/*    188 */	srl	%xg0,31,%xg2


/*    188 */	sxar2
/*    188 */	sra	%o0,%g0,%xg3
/*    188 */	add	%xg0,%xg2,%xg0


/*    188 */	sxar2
/*    188 */	sra	%xg1,%g0,%xg1
/*    188 */	sra	%xg0,1,%xg0


/*    188 */	sxar2
/*    188 */	add	%xg3,%xg3,%xg4
/*    188 */	add	%xg0,1,%xg0


/*    188 */	sxar2
/*    188 */	add	%xg1,%xg1,%xg5
/*    188 */	sra	%xg0,%g0,%xg0


/*    188 */	sxar2
/*    188 */	add	%xg4,%xg3,%xg4
/*    188 */	sub	%l5,%xg0,%xg0


/*    188 */	sxar2
/*    188 */	add	%xg5,%xg1,%xg5
/*    188 */	srax	%xg0,32,%xg6


/*    188 */	sxar2
/*    188 */	sllx	%xg4,5,%xg4
/*    188 */	and	%xg0,%xg6,%xg0


/*    188 */	sxar2
/*    188 */	sllx	%xg5,5,%xg5
/*    188 */	sub	%l6,%xg0,%xg0


/*    188 */	sxar2
/*    188 */	add	%xg8,%xg4,%xg4
/*    188 */	sub	%o0,%xg7,%xg7

/*    188 */	sxar1
/*    188 */	add	%xg8,%xg5,%xg8

.L6950:


/*     25 */	sxar2
/*     25 */	srl	%xg7,31,%xg9
/*    ??? */	ldd,s	[%fp+1775],%f106


/*    189 */	sxar2
/*    189 */	add	%xg9,%xg7,%xg9
/*    189 */	sra	%xg9,1,%xg9


/*     25 */	sxar2
/*     25 */	std,s	%f106,[%fp+1791]
/*     25 */	std,s	%f106,[%fp+1807]


/*     25 */	sxar2
/*     25 */	std,s	%f106,[%fp+1823]
/*     25 */	std,s	%f106,[%fp+1839]


/*     25 */	sxar2
/*     25 */	std,s	%f106,[%fp+1855]
/*     25 */	std,s	%f106,[%fp+1871]


/*     25 */	sxar2
/*     25 */	std,s	%f106,[%fp+1887]
/*     25 */	std,s	%f106,[%fp+1903]


/*     25 */	sxar2
/*     25 */	std,s	%f106,[%fp+1919]
/*     25 */	std,s	%f106,[%fp+1935]


/*     25 */	sxar2
/*     25 */	std,s	%f106,[%fp+1951]
/*     25 */	std,s	%f106,[%fp+1967]

.L6951:

/*    194 */	sxar1
/*    194 */	cmp	%xg11,%g0

/*    194 */	ble	.L6955
	nop


.L6952:


/*    194 */	sxar2
/*    194 */	sra	%xg9,%g0,%xg9
/*    ??? */	ldd,s	[%fp+1775],%f76


/*    194 */	sxar2
/*    194 */	sub	%xg11,2,%xg10
/*    194 */	add	%xg9,%xg9,%xg12


/*    194 */	sxar2
/*    194 */	cmp	%xg10,%g0
/*    194 */	add	%xg12,%xg9,%xg12


/*    194 */	sxar2
/*    194 */	sllx	%xg12,6,%xg12
/*    194 */	fmovd,s	%f76,%f72


/*    194 */	sxar2
/*    194 */	fmovd,s	%f72,%f68
/*    194 */	fmovd,s	%f72,%f64


/*    194 */	sxar2
/*    194 */	fmovd,s	%f72,%f60
/*    194 */	fmovd,s	%f72,%f56


/*    194 */	sxar2
/*    194 */	fmovd,s	%f72,%f52
/*    194 */	fmovd,s	%f72,%f48


/*    194 */	sxar2
/*    194 */	fmovd,s	%f72,%f44
/*    194 */	fmovd,s	%f72,%f40


/*    194 */	sxar2
/*    194 */	fmovd,s	%f72,%f36
/*    194 */	fmovd,s	%f72,%f32

/*    194 */	bl	.L6961
	nop


.L6964:


/*    208 */	sxar2
/*    208 */	fzero,s	%f80
/*    208 */	add	%l1,%xg12,%xg13


/*    194 */	sxar2
/*    194 */	fmovd,s	%f72,%f68
/*    194 */	cmp	%xg10,8


/*    208 */	sxar2
/*    208 */	add	%l7,%xg13,%xg14
/*    208 */	fmovd,s	%f72,%f64


/*    208 */	sxar2
/*    208 */	fmovd,s	%f72,%f60
/*    208 */	fmovd,s	%f72,%f56


/*    208 */	sxar2
/*    208 */	fmovd,s	%f72,%f52
/*    208 */	fmovd,s	%f72,%f48


/*    208 */	sxar2
/*    208 */	fmovd,s	%f72,%f44
/*    208 */	fmovd,s	%f72,%f40


/*    208 */	sxar2
/*    208 */	fmovd,s	%f72,%f36
/*    208 */	fmovd,s	%f72,%f32


/*    208 */	sxar2
/*    208 */	fmovd,s	%f80,%f88
/*    208 */	fmovd,s	%f80,%f84


/*    208 */	sxar2
/*    208 */	fmovd,s	%f88,%f92
/*    208 */	fmovd,s	%f88,%f96


/*    208 */	sxar2
/*    208 */	fmovd,s	%f88,%f100
/*    208 */	fmovd,s	%f88,%f104


/*    208 */	sxar2
/*    208 */	fmovd,s	%f88,%f108
/*    208 */	fmovd,s	%f88,%f112


/*    208 */	sxar2
/*    208 */	fmovd,s	%f88,%f116
/*    208 */	fmovd,s	%f88,%f120

/*    208 */	sxar1
/*    208 */	fmovd,s	%f88,%f124

/*    194 */	bl	.L7094
	nop


.L7090:


.L7097:


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg13+16],%f38
/*    194 */	ldd,s	[%xg13],%f34


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg13+48],%f46
/*    194 */	ldd,s	[%xg13+32],%f42


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg13+64],%f50
/*    194 */	ldd,s	[%xg13+96],%f58

/*    194 */	sxar1
/*    194 */	ldd,s	[%xg13+80],%f54

.L6953:


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg13+112],%f128
/*    194 */	faddd,s	%f32,%f34,%f32


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg13+128],%f130
/*    194 */	faddd,s	%f36,%f38,%f36


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg13+144],%f132
/*    194 */	faddd,s	%f40,%f42,%f40


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg13+160],%f134
/*    194 */	faddd,s	%f44,%f46,%f44


/*    194 */	sxar2
/*    194 */	faddd,s	%f48,%f50,%f48
/*    194 */	ldd,s	[%xg13+176],%f136


/*    194 */	sxar2
/*    194 */	faddd,s	%f52,%f54,%f52
/*    194 */	faddd,s	%f56,%f58,%f56


/*    194 */	sxar2
/*    194 */	faddd,s	%f60,%f128,%f60
/*    194 */	faddd,s	%f64,%f130,%f64


/*    194 */	sxar2
/*    194 */	faddd,s	%f68,%f132,%f68
/*    194 */	faddd,s	%f72,%f134,%f72


/*    194 */	sxar2
/*    194 */	faddd,s	%f76,%f136,%f76
/*    194 */	ldd,s	[%xg14+-4032],%f138


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg14+-4016],%f140
/*    194 */	ldd,s	[%xg14+-4000],%f142


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg14+-3984],%f144
/*    194 */	ldd,s	[%xg14+-3968],%f146


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg14+-3952],%f148
/*    194 */	ldd,s	[%xg14+-3936],%f150


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg14+-3920],%f152
/*    194 */	ldd,s	[%xg14+-3904],%f154


/*    194 */	sxar2
/*    194 */	faddd,s	%f80,%f138,%f80
/*    194 */	ldd,s	[%xg14+-3888],%f156


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg14+-3872],%f158
/*    194 */	faddd,s	%f84,%f140,%f84


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg14+-3856],%f160
/*    194 */	faddd,s	%f88,%f142,%f88


/*    194 */	sxar2
/*    194 */	faddd,s	%f92,%f144,%f92
/*    194 */	faddd,s	%f96,%f146,%f96


/*    194 */	sxar2
/*    194 */	faddd,s	%f100,%f148,%f100
/*    194 */	faddd,s	%f104,%f150,%f104


/*    194 */	sxar2
/*    194 */	faddd,s	%f108,%f152,%f108
/*    194 */	faddd,s	%f112,%f154,%f112


/*    194 */	sxar2
/*    194 */	faddd,s	%f116,%f156,%f116
/*    194 */	faddd,s	%f120,%f158,%f120


/*    194 */	sxar2
/*    194 */	faddd,s	%f124,%f160,%f124
/*    194 */	add	%i2,%xg13,%xg13


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg13+16],%f164
/*    194 */	ldd,s	[%xg13],%f162


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg13+48],%f168
/*    194 */	ldd,s	[%xg13+32],%f166


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg13+64],%f170
/*    194 */	ldd,s	[%xg13+96],%f174


/*    194 */	sxar2
/*    194 */	add	%i2,%xg12,%xg12
/*    194 */	ldd,s	[%xg13+80],%f172


/*    194 */	sxar2
/*    194 */	add	%i2,%xg14,%xg14
/*    194 */	ldd,s	[%xg13+112],%f176


/*    194 */	sxar2
/*    194 */	faddd,s	%f32,%f162,%f32
/*    194 */	ldd,s	[%xg13+128],%f178


/*    194 */	sxar2
/*    194 */	faddd,s	%f36,%f164,%f36
/*    194 */	ldd,s	[%xg13+144],%f180


/*    194 */	sxar2
/*    194 */	faddd,s	%f40,%f166,%f40
/*    194 */	ldd,s	[%xg13+160],%f182


/*    194 */	sxar2
/*    194 */	faddd,s	%f44,%f168,%f44
/*    194 */	faddd,s	%f48,%f170,%f48


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg13+176],%f184
/*    194 */	faddd,s	%f52,%f172,%f52


/*    194 */	sxar2
/*    194 */	faddd,s	%f56,%f174,%f56
/*    194 */	faddd,s	%f60,%f176,%f60


/*    194 */	sxar2
/*    194 */	faddd,s	%f64,%f178,%f64
/*    194 */	faddd,s	%f68,%f180,%f68


/*    194 */	sxar2
/*    194 */	faddd,s	%f72,%f182,%f72
/*    194 */	faddd,s	%f76,%f184,%f76


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg14+-4032],%f186
/*    194 */	ldd,s	[%xg14+-4016],%f188


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg14+-4000],%f190
/*    194 */	ldd,s	[%xg14+-3984],%f192


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg14+-3968],%f194
/*    194 */	ldd,s	[%xg14+-3952],%f196


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg14+-3936],%f198
/*    194 */	ldd,s	[%xg14+-3920],%f200


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg14+-3904],%f202
/*    194 */	faddd,s	%f80,%f186,%f80


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg14+-3888],%f204
/*    194 */	ldd,s	[%xg14+-3872],%f206


/*    194 */	sxar2
/*    194 */	faddd,s	%f84,%f188,%f84
/*    194 */	ldd,s	[%xg14+-3856],%f208


/*    194 */	sxar2
/*    194 */	faddd,s	%f88,%f190,%f88
/*    194 */	faddd,s	%f92,%f192,%f92


/*    194 */	sxar2
/*    194 */	faddd,s	%f96,%f194,%f96
/*    194 */	faddd,s	%f100,%f196,%f100


/*    194 */	sxar2
/*    194 */	faddd,s	%f104,%f198,%f104
/*    194 */	faddd,s	%f108,%f200,%f108


/*    194 */	sxar2
/*    194 */	faddd,s	%f112,%f202,%f112
/*    194 */	faddd,s	%f116,%f204,%f116


/*    194 */	sxar2
/*    194 */	faddd,s	%f120,%f206,%f120
/*    194 */	faddd,s	%f124,%f208,%f124


/*    194 */	sxar2
/*    194 */	add	%i2,%xg13,%xg13
/*    194 */	ldd,s	[%xg13+16],%f38


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg13],%f34
/*    194 */	ldd,s	[%xg13+48],%f46


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg13+32],%f42
/*    194 */	ldd,s	[%xg13+64],%f50


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg13+96],%f58
/*    194 */	add	%i2,%xg12,%xg12


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg13+80],%f54
/*    194 */	add	%i2,%xg14,%xg14


/*    194 */	sxar2
/*    194 */	sub	%xg10,4,%xg10
/*    194 */	cmp	%xg10,9

/*    194 */	bge,pt	%icc, .L6953
	nop


.L7098:


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg13+112],%f62
/*    194 */	faddd,s	%f32,%f34,%f32


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg13+128],%f66
/*    194 */	faddd,s	%f36,%f38,%f36


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg13+144],%f70
/*    194 */	faddd,s	%f40,%f42,%f40


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg13+160],%f74
/*    194 */	faddd,s	%f44,%f46,%f44


/*    194 */	sxar2
/*    194 */	faddd,s	%f48,%f50,%f48
/*    194 */	ldd,s	[%xg13+176],%f78


/*    194 */	sxar2
/*    194 */	faddd,s	%f52,%f54,%f52
/*    194 */	ldd,s	[%xg14+-4032],%f82


/*    194 */	sxar2
/*    194 */	faddd,s	%f56,%f58,%f56
/*    194 */	ldd,s	[%xg14+-4016],%f86


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg14+-4000],%f90
/*    194 */	add	%i2,%xg13,%xg13


/*    194 */	sxar2
/*    194 */	faddd,s	%f60,%f62,%f60
/*    194 */	faddd,s	%f64,%f66,%f64


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg14+-3984],%f94
/*    194 */	ldd,s	[%xg14+-3968],%f98


/*    194 */	sxar2
/*    194 */	faddd,s	%f68,%f70,%f68
/*    194 */	faddd,s	%f72,%f74,%f72


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg14+-3952],%f102
/*    194 */	ldd,s	[%xg14+-3936],%f106


/*    194 */	sxar2
/*    194 */	faddd,s	%f76,%f78,%f76
/*    194 */	ldd,s	[%xg14+-3920],%f110


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg14+-3904],%f114
/*    194 */	faddd,s	%f80,%f82,%f80


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg14+-3888],%f118
/*    194 */	ldd,s	[%xg14+-3872],%f122


/*    194 */	sxar2
/*    194 */	faddd,s	%f84,%f86,%f84
/*    194 */	faddd,s	%f88,%f90,%f88


/*    194 */	sxar2
/*    194 */	ldd,s	[%xg14+-3856],%f126
/*    194 */	faddd,s	%f92,%f94,%f92


/*    194 */	sxar2
/*    194 */	faddd,s	%f96,%f98,%f96
/*    194 */	add	%i2,%xg12,%xg12


/*    194 */	sxar2
/*    194 */	faddd,s	%f100,%f102,%f100
/*    194 */	faddd,s	%f104,%f106,%f104


/*    194 */	sxar2
/*    194 */	add	%i2,%xg14,%xg14
/*    194 */	sub	%xg10,2,%xg10


/*    194 */	sxar2
/*    194 */	faddd,s	%f108,%f110,%f108
/*    194 */	faddd,s	%f112,%f114,%f112


/*    194 */	sxar2
/*    194 */	faddd,s	%f116,%f118,%f116
/*    194 */	faddd,s	%f120,%f122,%f120

/*    194 */	sxar1
/*    194 */	faddd,s	%f124,%f126,%f124

.L7094:


.L7093:


.L7096:


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg13],%f210
/*     43 */	ldd,s	[%xg13+16],%f212


/*    208 */	sxar2
/*    208 */	add	%i2,%xg12,%xg12
/*    208 */	subcc	%xg10,2,%xg10


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg13+32],%f214
/*     43 */	ldd,s	[%xg13+48],%f216


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg13+64],%f218
/*     43 */	ldd,s	[%xg13+80],%f220


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg13+96],%f222
/*     43 */	ldd,s	[%xg13+112],%f224


/*     43 */	sxar2
/*     43 */	faddd,s	%f32,%f210,%f32
/*     43 */	faddd,s	%f36,%f212,%f36


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg13+128],%f226
/*     43 */	ldd,s	[%xg13+144],%f228


/*     43 */	sxar2
/*     43 */	faddd,s	%f40,%f214,%f40
/*     43 */	faddd,s	%f44,%f216,%f44


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg13+160],%f230
/*     43 */	ldd,s	[%xg13+176],%f232


/*     43 */	sxar2
/*     43 */	faddd,s	%f48,%f218,%f48
/*     43 */	faddd,s	%f52,%f220,%f52


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg14+-4032],%f234
/*     43 */	ldd,s	[%xg14+-4016],%f236


/*     43 */	sxar2
/*     43 */	faddd,s	%f56,%f222,%f56
/*     43 */	faddd,s	%f60,%f224,%f60


/*    208 */	sxar2
/*    208 */	ldd,s	[%xg14+-4000],%f238
/*    208 */	add	%i2,%xg13,%xg13


/*     43 */	sxar2
/*     43 */	faddd,s	%f64,%f226,%f64
/*     43 */	faddd,s	%f68,%f228,%f68


/*     43 */	sxar2
/*     43 */	faddd,s	%f72,%f230,%f72
/*     43 */	faddd,s	%f76,%f232,%f76


/*     43 */	sxar2
/*     43 */	faddd,s	%f80,%f234,%f80
/*     43 */	faddd,s	%f84,%f236,%f84


/*     43 */	sxar2
/*     43 */	faddd,s	%f88,%f238,%f88
/*     43 */	ldd,s	[%xg14+-3984],%f240


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg14+-3968],%f242
/*     43 */	ldd,s	[%xg14+-3952],%f244


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg14+-3936],%f246
/*     43 */	ldd,s	[%xg14+-3920],%f248


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg14+-3904],%f250
/*     43 */	ldd,s	[%xg14+-3888],%f252


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg14+-3872],%f254
/*     43 */	faddd,s	%f92,%f240,%f92


/*     43 */	sxar2
/*     43 */	faddd,s	%f96,%f242,%f96
/*     43 */	ldd,s	[%xg14+-3856],%f34


/*     43 */	sxar2
/*     43 */	add	%i2,%xg14,%xg14
/*     43 */	faddd,s	%f100,%f244,%f100


/*     43 */	sxar2
/*     43 */	faddd,s	%f104,%f246,%f104
/*     43 */	faddd,s	%f108,%f248,%f108


/*     43 */	sxar2
/*     43 */	faddd,s	%f112,%f250,%f112
/*     43 */	faddd,s	%f116,%f252,%f116


/*     43 */	sxar2
/*     43 */	faddd,s	%f120,%f254,%f120
/*     43 */	faddd,s	%f124,%f34,%f124

/*    208 */	bpos,pt	%icc, .L7096
	nop


.L7092:


/*    208 */	sxar2
/*    208 */	faddd,s	%f76,%f124,%f76
/*    208 */	faddd,s	%f72,%f120,%f72


/*    208 */	sxar2
/*    208 */	faddd,s	%f68,%f116,%f68
/*    208 */	faddd,s	%f64,%f112,%f64


/*    208 */	sxar2
/*    208 */	faddd,s	%f60,%f108,%f60
/*    208 */	faddd,s	%f56,%f104,%f56


/*    208 */	sxar2
/*    208 */	faddd,s	%f52,%f100,%f52
/*    208 */	faddd,s	%f48,%f96,%f48


/*    208 */	sxar2
/*    208 */	faddd,s	%f44,%f92,%f44
/*    208 */	faddd,s	%f40,%f88,%f40


/*    208 */	sxar2
/*    208 */	faddd,s	%f36,%f84,%f36
/*    208 */	faddd,s	%f32,%f80,%f32

.L6961:

/*    194 */	sxar1
/*    194 */	addcc	%xg10,1,%xg10

/*    194 */	bneg	.L6954
	nop


.L6962:

/*    194 */	sxar1
/*    194 */	add	%l1,%xg12,%xg12

.L6967:


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg12],%f38
/*     43 */	ldd,s	[%xg12+16],%f42


/*     43 */	sxar2
/*     43 */	subcc	%xg10,1,%xg10
/*     43 */	ldd,s	[%xg12+32],%f46


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg12+48],%f50
/*     43 */	ldd,s	[%xg12+64],%f54


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg12+80],%f58
/*     43 */	ldd,s	[%xg12+96],%f62


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg12+112],%f66
/*     43 */	faddd,s	%f32,%f38,%f32


/*     43 */	sxar2
/*     43 */	faddd,s	%f36,%f42,%f36
/*     43 */	ldd,s	[%xg12+128],%f70


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg12+144],%f74
/*     43 */	faddd,s	%f40,%f46,%f40


/*     43 */	sxar2
/*     43 */	faddd,s	%f44,%f50,%f44
/*     43 */	ldd,s	[%xg12+160],%f78


/*     43 */	sxar2
/*     43 */	ldd,s	[%xg12+176],%f80
/*     43 */	faddd,s	%f48,%f54,%f48


/*    208 */	sxar2
/*    208 */	faddd,s	%f52,%f58,%f52
/*    208 */	add	%i1,%xg12,%xg12


/*     43 */	sxar2
/*     43 */	faddd,s	%f56,%f62,%f56
/*     43 */	faddd,s	%f60,%f66,%f60


/*     43 */	sxar2
/*     43 */	faddd,s	%f64,%f70,%f64
/*     43 */	faddd,s	%f68,%f74,%f68


/*     43 */	sxar2
/*     43 */	faddd,s	%f72,%f78,%f72
/*     43 */	faddd,s	%f76,%f80,%f76

/*    208 */	bpos,pt	%icc, .L6967
	nop


.L6963:


.L6954:


/*    208 */	sxar2
/*    208 */	std,s	%f32,[%fp+1791]
/*    208 */	std,s	%f36,[%fp+1807]


/*    208 */	sxar2
/*    208 */	std,s	%f40,[%fp+1823]
/*    208 */	std,s	%f44,[%fp+1839]


/*    208 */	sxar2
/*    208 */	std,s	%f48,[%fp+1855]
/*    208 */	std,s	%f52,[%fp+1871]


/*    208 */	sxar2
/*    208 */	std,s	%f56,[%fp+1887]
/*    208 */	std,s	%f60,[%fp+1903]


/*    208 */	sxar2
/*    208 */	std,s	%f64,[%fp+1919]
/*    208 */	std,s	%f68,[%fp+1935]


/*    208 */	sxar2
/*    208 */	std,s	%f72,[%fp+1951]
/*    208 */	std,s	%f76,[%fp+1967]

.L6955:



/*    234 */	sxar2
/*    234 */	ldd,s	[%fp+1791],%f82
/*    234 */	add	%xg7,2,%xg7



/*     81 */	sxar2
/*     81 */	subcc	%xg0,1,%xg0
/*     81 */	std	%f82,[%xg4]



/*     81 */	sxar2
/*     81 */	ldd,s	[%fp+1807],%f84
/*     81 */	std	%f84,[%xg4+8]



/*     81 */	sxar2
/*     81 */	ldd,s	[%fp+1823],%f86
/*     81 */	std	%f86,[%xg4+16]



/*     81 */	sxar2
/*     81 */	ldd,s	[%fp+1839],%f88
/*     81 */	std	%f88,[%xg4+24]



/*     81 */	sxar2
/*     81 */	ldd,s	[%fp+1855],%f90
/*     81 */	std	%f90,[%xg4+32]



/*     81 */	sxar2
/*     81 */	ldd,s	[%fp+1871],%f92
/*     81 */	std	%f92,[%xg4+40]



/*     81 */	sxar2
/*     81 */	ldd,s	[%fp+1887],%f94
/*     81 */	std	%f94,[%xg4+48]



/*     81 */	sxar2
/*     81 */	ldd,s	[%fp+1903],%f96
/*     81 */	std	%f96,[%xg4+56]



/*     81 */	sxar2
/*     81 */	ldd,s	[%fp+1919],%f98
/*     81 */	std	%f98,[%xg4+64]



/*     81 */	sxar2
/*     81 */	ldd,s	[%fp+1935],%f100
/*     81 */	std	%f100,[%xg4+72]



/*     81 */	sxar2
/*     81 */	ldd,s	[%fp+1951],%f102
/*     81 */	std	%f102,[%xg4+80]



/*     81 */	sxar2
/*     81 */	ldd,s	[%fp+1967],%f104
/*     81 */	std	%f104,[%xg4+88]


/*     84 */	sxar2
/*     84 */	add	%xg4,192,%xg4
/*     84 */	std	%f338,[%xg8]


/*     84 */	sxar2
/*     84 */	std	%f340,[%xg8+8]
/*     84 */	std	%f342,[%xg8+16]


/*     84 */	sxar2
/*     84 */	std	%f344,[%xg8+24]
/*     84 */	std	%f346,[%xg8+32]


/*     84 */	sxar2
/*     84 */	std	%f348,[%xg8+40]
/*     84 */	std	%f350,[%xg8+48]


/*     84 */	sxar2
/*     84 */	std	%f352,[%xg8+56]
/*     84 */	std	%f354,[%xg8+64]


/*     84 */	sxar2
/*     84 */	std	%f356,[%xg8+72]
/*     84 */	std	%f358,[%xg8+80]


/*    234 */	sxar2
/*    234 */	std	%f360,[%xg8+88]
/*    234 */	add	%xg8,192,%xg8

/*    234 */	bne,pt	%icc, .L6950
/*    234 */	add	%o0,2,%o0


.L6956:

/*    234 */
/*    234 */	ba	.L6948
	nop


.L6958:

/*    234 *//*    234 */	call	__mpc_obar
/*    234 */	ldx	[%fp+2199],%o0

/*    234 *//*    234 */	call	__mpc_obar
/*    234 */	ldx	[%fp+2199],%o0


.L6959:

/*    234 */	ret
	restore



.LLFE8:
	.size	_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_3,.-_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_3
	.type	_ZN7Gravity19calc_force_in_rangeEiidP5Force._OMP_3,#function
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite8-k.cpp _ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd $"
	.section	".text"
	.global	_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd
	.align	64
_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd:
.LLFB9:
.L633:

/*    237 */	save	%sp,-880,%sp
.LLCFI7:
/*    237 */	stw	%i0,[%fp+2179]
/*    237 */	std	%f2,[%fp+2183]
/*    237 */	stx	%i2,[%fp+2191]
/*    237 */	stx	%i3,[%fp+2199]
/*    237 */	stx	%i4,[%fp+2207]

.L634:

/*    245 *//*    245 */	sethi	%h44(_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_4),%o0
/*    245 */	mov	%fp,%l0
/*    245 */	or	%o0,%m44(_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_4),%o0
/*    245 */	mov	%g0,%l1
/*    245 */	sllx	%o0,12,%o0
/*    245 */	mov	%l0,%o1
/*    245 */	or	%o0,%l44(_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_4),%o0
/*    245 */	call	__mpc_opar
/*    245 */	mov	%l1,%o2

/*    255 */
/*    257 *//*    257 */	sethi	%h44(_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_5),%o0
/*    257 */	mov	%l0,%o1
/*    257 */	or	%o0,%m44(_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_5),%o0
/*    257 */	mov	%l1,%o2
/*    257 */	sllx	%o0,12,%o0
/*    257 */	call	__mpc_opar
/*    257 */	or	%o0,%l44(_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_5),%o0

/*    279 */
/*    279 */	ret
	restore



.L678:


.LLFE9:
	.size	_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd,.-_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd
	.type	_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd,#function
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite8-k.cpp _ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_4 $"
	.section	".text"
	.align	64
_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_4:
.LLFB10:
.L6969:

/*    245 */	save	%sp,-640,%sp
.LLCFI8:
/*    245 */	stx	%i0,[%fp+2175]
/*    245 */	stx	%i3,[%fp+2199]
/*    245 */	stx	%i0,[%fp+2175]

.L6970:

/*    245 *//*    245 */	sxar1
/*    245 */	ldsw	[%i0+2031],%xg11
/*    245 */
/*    245 */
/*    245 */
/*    246 */	ldsw	[%i0+2179],%l0
/*    246 */	cmp	%l0,%g0
/*    246 */	ble	.L6977
/*    246 */	mov	%g0,%o0


.L6971:

/*    246 */	sxar1
/*    246 */	mov	1,%xg10

/*    246 */	mov	1,%l5

/*    246 */	sxar1
/*    246 */	stx	%xg10,[%fp+2031]

/*    246 */	sra	%l0,%g0,%l0

/*    246 */	add	%fp,2039,%l1

/*    246 */	add	%fp,2023,%l2

/*    246 */	add	%fp,2031,%l3

/*    246 */	sra	%l5,%g0,%l4

.L6972:

/*    246 */	sra	%o0,%g0,%o0

/*    246 */	stx	%g0,[%sp+2223]

/*    246 */	mov	2,%o2

/*    246 */	mov	%g0,%o3

/*    246 */	mov	%l0,%o1

/*    246 */	mov	%l1,%o4


/*    246 */	stx	%g0,[%sp+2231]

/*    246 */	stx	%l3,[%sp+2239]


/*    246 */	sxar2
/*    246 */	ldx	[%fp+2199],%xg8
/*    246 */	stx	%xg8,[%sp+2247]

/*    246 */	call	__mpc_ostd_th
/*    246 */	mov	%l2,%o5
/*    246 */	sxar2
/*    246 */	ldx	[%fp+2031],%xg9
/*    246 */	cmp	%xg9,%g0
/*    246 */	ble,pn	%xcc, .L6977
	nop


.L6973:

/*    246 */	ldx	[%fp+2039],%o0

/*    246 */	sxar1
/*    246 */	ldx	[%fp+2023],%xg0

/*    246 */	sra	%o0,%g0,%o0


/*    246 */	sxar2
/*    246 */	sra	%xg0,%g0,%xg0
/*    246 */	sub	%xg0,%o0,%xg0


/*    246 */	sxar2
/*    246 */	add	%o0,1,%xg1
/*    246 */	srl	%xg0,31,%xg2


/*    246 */	sxar2
/*    246 */	sra	%o0,%g0,%xg3
/*    246 */	add	%xg0,%xg2,%xg0


/*    246 */	sxar2
/*    246 */	sra	%xg1,%g0,%xg1
/*    246 */	sra	%xg0,1,%xg0


/*    246 */	sxar2
/*    246 */	sllx	%xg3,5,%xg3
/*    246 */	add	%xg0,1,%xg0


/*    246 */	sxar2
/*    246 */	sllx	%xg1,5,%xg1
/*    246 */	sra	%xg0,%g0,%xg0


/*    246 */	sxar2
/*    246 */	sub	%l4,%xg0,%xg0
/*    246 */	srax	%xg0,32,%xg4


/*    246 */	sxar2
/*    246 */	and	%xg0,%xg4,%xg0
/*    246 */	sub	%l5,%xg0,%xg0

/*    246 */	sxar1
/*    246 */	subcc	%xg0,4,%xg0

/*    246 */	bneg	.L6980
	nop


.L6983:


/*    246 */	sxar2
/*    246 */	ldx	[%i0+2191],%xg5
/*    246 */	ldx	[%i0+2199],%xg10


/*    246 */	sxar2
/*    246 */	cmp	%xg0,20
/*    246 */	add	%xg5,16,%xg6


/*    246 */	sxar2
/*    246 */	add	%xg10,%xg3,%xg9
/*    246 */	add	%xg5,32,%xg7


/*    246 */	sxar2
/*    246 */	add	%xg5,48,%xg8
/*    246 */	add	%xg10,%xg1,%xg10

/*    246 */	bl	.L7103
	nop


.L7099:


.L7106:


/*    246 */	sxar2
/*    246 */	srl	%o0,31,%xg11
/*    246 */	add	%o0,2,%xg12


/*    246 */	sxar2
/*    246 */	add	%xg11,%o0,%xg11
/*    246 */	add	%o0,8,%xg13


/*    246 */	sxar2
/*    246 */	sra	%xg11,1,%xg11
/*    246 */	srl	%xg12,31,%xg14


/*    246 */	sxar2
/*    246 */	sra	%xg11,%g0,%xg11
/*    246 */	srl	%xg13,31,%xg15


/*    246 */	sxar2
/*    246 */	mulx	%xg11,416,%xg11
/*    246 */	add	%xg12,%xg14,%xg12


/*    246 */	sxar2
/*    246 */	sra	%xg12,1,%xg12
/*    246 */	add	%xg15,%xg13,%xg15


/*    246 */	sxar2
/*    246 */	sra	%xg12,%g0,%xg12
/*    246 */	sra	%xg15,1,%xg15


/*    246 */	sxar2
/*    246 */	add	%o0,4,%xg16
/*    246 */	sra	%xg15,%g0,%xg15


/*    246 */	sxar2
/*    246 */	add	%o0,10,%xg17
/*    246 */	srl	%xg16,31,%xg18


/*    246 */	sxar2
/*    246 */	add	%o0,6,%xg19
/*    246 */	srl	%xg17,31,%xg20


/*    246 */	sxar2
/*    246 */	add	%xg11,%xg5,%xg21
/*    246 */	mulx	%xg12,416,%xg12


/*    246 */	sxar2
/*    246 */	add	%xg11,%xg6,%xg22
/*    246 */	ldd,s	[%xg21],%f34


/*    246 */	sxar2
/*    246 */	add	%xg16,%xg18,%xg16
/*    246 */	ldd,s	[%xg22],%f32


/*    246 */	sxar2
/*    246 */	add	%xg11,%xg7,%xg23
/*    246 */	mulx	%xg15,416,%xg15

.L6974:


/*    246 */	sxar2
/*    246 */	add	%xg11,%xg8,%xg11
/*    246 */	sra	%xg16,1,%xg16


/*    246 */	sxar2
/*    246 */	ldd,s	[%xg23],%f130
/*    246 */	add	%xg15,%xg5,%xg21


/*    246 */	sxar2
/*    246 */	ldd,s	[%xg11],%f134
/*    246 */	sra	%xg16,%g0,%xg16


/*    246 */	sxar2
/*    246 */	mulx	%xg16,416,%xg16
/*    246 */	add	%xg12,%xg5,%xg22



/*    246 */	sxar2
/*    246 */	add	%xg12,%xg6,%xg23
/*    246 */	fmovd	%f290,%f128



/*    246 */	sxar2
/*    246 */	fmovd	%f288,%f384
/*    246 */	fmovd	%f32,%f290


/*    246 */	sxar2
/*    246 */	ldd,s	[%xg22],%f136
/*    246 */	add	%xg15,%xg6,%xg24


/*    246 */	sxar2
/*    246 */	ldd,s	[%xg23],%f140
/*    246 */	srl	%xg19,31,%xg25



/*    246 */	sxar2
/*    246 */	ldd,s	[%xg21],%f172
/*    246 */	ldd,s	[%xg24],%f176


/*    246 */	sxar2
/*    246 */	add	%xg12,%xg7,%xg26
/*    246 */	add	%xg19,%xg25,%xg19




/*    246 */	sxar2
/*    246 */	fmovd	%f386,%f132
/*    246 */	fmovd	%f390,%f388


/*    246 */	sxar2
/*    246 */	fmovd	%f134,%f386
/*    246 */	std,s	%f34,[%xg9]


/*    246 */	sxar2
/*    246 */	add	%xg12,%xg8,%xg12
/*    246 */	sra	%xg19,1,%xg19



/*    246 */	sxar2
/*    246 */	ldd,s	[%xg26],%f142
/*    246 */	add	%xg17,%xg20,%xg20


/*    246 */	sxar2
/*    246 */	ldd,s	[%xg12],%f146
/*    246 */	sra	%xg19,%g0,%xg19



/*    246 */	sxar2
/*    246 */	std,s	%f130,[%xg9+16]
/*    246 */	mulx	%xg19,416,%xg19




/*    246 */	sxar2
/*    246 */	sra	%xg20,1,%xg20
/*    246 */	fmovd	%f392,%f138



/*    246 */	sxar2
/*    246 */	fmovd	%f396,%f394
/*    246 */	fmovd	%f140,%f392


/*    246 */	sxar2
/*    246 */	std,s	%f128,[%xg10]
/*    246 */	add	%xg17,2,%xg27



/*    246 */	sxar2
/*    246 */	sra	%xg20,%g0,%xg20
/*    246 */	std,s	%f132,[%xg10+16]


/*    246 */	sxar2
/*    246 */	add	%xg17,4,%xg28
/*    246 */	add	%xg16,%xg5,%xg29


/*    246 */	sxar2
/*    246 */	std,s	%f136,[%xg9+64]
/*    246 */	add	%xg16,%xg6,%xg30




/*    246 */	sxar2
/*    246 */	add	%xg16,%xg7,%xg31
/*    246 */	ldd,s	[%xg29],%f148


/*    246 */	sxar2
/*    246 */	add	%xg16,%xg8,%xg16
/*    246 */	ldd,s	[%xg30],%f152


/*    246 */	sxar2
/*    246 */	fmovd	%f142,%f144
/*    246 */	fmovd	%f146,%f400



/*    246 */	sxar2
/*    246 */	ldd,s	[%xg31],%f154
/*    246 */	add	%xg17,6,%g1



/*    246 */	sxar2
/*    246 */	ldd,s	[%xg16],%f158
/*    246 */	std,s	%f144,[%xg9+80]


/*    246 */	sxar2
/*    246 */	add	%xg19,%xg5,%g2
/*    246 */	std,s	%f138,[%xg10+64]


/*    246 */	sxar2
/*    246 */	add	%xg19,%xg6,%g3
/*    246 */	add	%xg19,%xg7,%g4


/*    246 */	sxar2
/*    246 */	ldd,s	[%g2],%f160
/*    246 */	mulx	%xg20,416,%xg20


/*    246 */	sxar2
/*    246 */	add	%xg19,%xg8,%xg19
/*    246 */	ldd,s	[%g3],%f164




/*    246 */	sxar2
/*    246 */	fmovd	%f398,%f146
/*    246 */	fmovd	%f404,%f150



/*    246 */	sxar2
/*    246 */	fmovd	%f408,%f406
/*    246 */	ldd,s	[%g4],%f166


/*    246 */	sxar2
/*    246 */	ldd,s	[%xg19],%f170
/*    246 */	fmovd	%f152,%f404





/*    246 */	sxar2
/*    246 */	fmovd	%f410,%f156
/*    246 */	fmovd	%f414,%f412


/*    246 */	sxar2
/*    246 */	fmovd	%f158,%f410
/*    246 */	std,s	%f146,[%xg10+80]



/*    246 */	sxar2
/*    246 */	srl	%g1,31,%xg11
/*    246 */	std,s	%f148,[%xg9+128]



/*    246 */	sxar2
/*    246 */	add	%xg11,%g1,%xg11
/*    246 */	std,s	%f154,[%xg9+144]





/*    246 */	sxar2
/*    246 */	sra	%xg11,1,%xg11
/*    246 */	std,s	%f150,[%xg10+128]


/*    246 */	sxar2
/*    246 */	add	%xg17,8,%xg12
/*    246 */	sra	%xg11,%g0,%xg11


/*    246 */	sxar2
/*    246 */	fmovd	%f160,%f162
/*    246 */	fmovd	%f164,%f418





/*    246 */	sxar2
/*    246 */	std,s	%f156,[%xg10+144]
/*    246 */	mulx	%xg11,416,%xg11


/*    246 */	sxar2
/*    246 */	srl	%xg27,31,%g5
/*    246 */	fmovd	%f166,%f168



/*    246 */	sxar2
/*    246 */	fmovd	%f170,%f424
/*    246 */	std,s	%f162,[%xg9+192]


/*    246 */	sxar2
/*    246 */	srl	%xg12,31,%o0
/*    246 */	add	%xg27,%g5,%xg27



/*    246 */	sxar2
/*    246 */	fmovd	%f416,%f164
/*    246 */	std,s	%f168,[%xg9+208]


/*    246 */	sxar2
/*    246 */	add	%xg15,%xg7,%o1
/*    246 */	fmovd	%f422,%f170



/*    246 */	sxar2
/*    246 */	std,s	%f164,[%xg10+192]
/*    246 */	std,s	%f170,[%xg10+208]


/*    246 */	sxar2
/*    246 */	add	%xg15,%xg8,%xg15
/*    246 */	sra	%xg27,1,%xg27


/*    246 */	sxar2
/*    246 */	ldd,s	[%o1],%f178
/*    246 */	add	%xg11,%xg5,%o2


/*    246 */	sxar2
/*    246 */	ldd,s	[%xg15],%f182
/*    246 */	sra	%xg27,%g0,%xg27


/*    246 */	sxar2
/*    246 */	mulx	%xg27,416,%xg27
/*    246 */	add	%xg20,%xg5,%o3




/*    246 */	sxar2
/*    246 */	add	%xg20,%xg6,%o4
/*    246 */	ldd,s	[%o3],%f184


/*    246 */	sxar2
/*    246 */	add	%xg11,%xg6,%o5
/*    246 */	ldd,s	[%o4],%f188


/*    246 */	sxar2
/*    246 */	srl	%xg28,31,%o7
/*    246 */	fmovd	%f172,%f174



/*    246 */	sxar2
/*    246 */	fmovd	%f176,%f430
/*    246 */	ldd,s	[%o2],%f34


/*    246 */	sxar2
/*    246 */	ldd,s	[%o5],%f32
/*    246 */	add	%xg20,%xg7,%xg2




/*    246 */	sxar2
/*    246 */	add	%xg28,%o7,%xg28
/*    246 */	std,s	%f174,[%xg9+256]


/*    246 */	sxar2
/*    246 */	add	%xg20,%xg8,%xg20
/*    246 */	sra	%xg28,1,%xg28


/*    246 */	sxar2
/*    246 */	fmovd	%f178,%f180
/*    246 */	fmovd	%f182,%f436



/*    246 */	sxar2
/*    246 */	ldd,s	[%xg2],%f190
/*    246 */	add	%xg12,%o0,%xg12


/*    246 */	sxar2
/*    246 */	ldd,s	[%xg20],%f194
/*    246 */	sra	%xg28,%g0,%xg28



/*    246 */	sxar2
/*    246 */	fmovd	%f428,%f176
/*    246 */	std,s	%f180,[%xg9+272]


/*    246 */	sxar2
/*    246 */	mulx	%xg28,416,%xg28
/*    246 */	sra	%xg12,1,%xg12




/*    246 */	sxar2
/*    246 */	fmovd	%f434,%f182
/*    246 */	fmovd	%f440,%f186



/*    246 */	sxar2
/*    246 */	fmovd	%f444,%f442
/*    246 */	fmovd	%f188,%f440


/*    246 */	sxar2
/*    246 */	std,s	%f176,[%xg10+256]
/*    246 */	add	%xg17,10,%xg16



/*    246 */	sxar2
/*    246 */	sra	%xg12,%g0,%xg12
/*    246 */	std,s	%f182,[%xg10+272]


/*    246 */	sxar2
/*    246 */	add	%xg17,12,%xg19
/*    246 */	add	%xg27,%xg5,%xg4


/*    246 */	sxar2
/*    246 */	std,s	%f184,[%xg9+320]
/*    246 */	add	%xg27,%xg6,%xg13




/*    246 */	sxar2
/*    246 */	add	%xg27,%xg7,%xg14
/*    246 */	ldd,s	[%xg4],%f196


/*    246 */	sxar2
/*    246 */	add	%xg27,%xg8,%xg27
/*    246 */	ldd,s	[%xg13],%f200


/*    246 */	sxar2
/*    246 */	fmovd	%f190,%f192
/*    246 */	fmovd	%f194,%f448



/*    246 */	sxar2
/*    246 */	ldd,s	[%xg14],%f202
/*    246 */	add	%xg17,14,%xg18



/*    246 */	sxar2
/*    246 */	ldd,s	[%xg27],%f206
/*    246 */	std,s	%f192,[%xg9+336]


/*    246 */	sxar2
/*    246 */	add	%xg28,%xg5,%xg20
/*    246 */	std,s	%f186,[%xg10+320]


/*    246 */	sxar2
/*    246 */	add	%xg28,%xg6,%xg21
/*    246 */	add	%xg28,%xg7,%xg22


/*    246 */	sxar2
/*    246 */	ldd,s	[%xg20],%f208
/*    246 */	mulx	%xg12,416,%xg12


/*    246 */	sxar2
/*    246 */	add	%xg28,%xg8,%xg28
/*    246 */	ldd,s	[%xg21],%f212





/*    246 */	sxar2
/*    246 */	fmovd	%f446,%f194
/*    246 */	ldd,s	[%xg22],%f214


/*    246 */	sxar2
/*    246 */	ldd,s	[%xg28],%f218
/*    246 */	fmovd	%f196,%f198





/*    246 */	sxar2
/*    246 */	fmovd	%f200,%f454
/*    246 */	std,s	%f194,[%xg10+336]


/*    246 */	sxar2
/*    246 */	srl	%xg18,31,%xg15
/*    246 */	fmovd	%f202,%f204



/*    246 */	sxar2
/*    246 */	fmovd	%f206,%f460
/*    246 */	std,s	%f198,[%xg9+384]


/*    246 */	sxar2
/*    246 */	add	%xg15,%xg18,%xg15
/*    246 */	fmovd	%f452,%f200



/*    246 */	sxar2
/*    246 */	std,s	%f204,[%xg9+400]
/*    246 */	sra	%xg15,1,%xg15





/*    246 */	sxar2
/*    246 */	fmovd	%f458,%f206
/*    246 */	std,s	%f200,[%xg10+384]


/*    246 */	sxar2
/*    246 */	add	%xg17,16,%xg17
/*    246 */	sra	%xg15,%g0,%xg15


/*    246 */	sxar2
/*    246 */	fmovd	%f208,%f210
/*    246 */	fmovd	%f212,%f466





/*    246 */	sxar2
/*    246 */	std,s	%f206,[%xg10+400]
/*    246 */	mulx	%xg15,416,%xg15


/*    246 */	sxar2
/*    246 */	srl	%xg16,31,%xg23
/*    246 */	fmovd	%f214,%f216



/*    246 */	sxar2
/*    246 */	fmovd	%f218,%f472
/*    246 */	std,s	%f210,[%xg9+448]


/*    246 */	sxar2
/*    246 */	srl	%xg17,31,%xg20
/*    246 */	add	%xg16,%xg23,%xg16



/*    246 */	sxar2
/*    246 */	fmovd	%f464,%f212
/*    246 */	std,s	%f216,[%xg9+464]


/*    246 */	sxar2
/*    246 */	add	%xg11,%xg7,%xg23
/*    246 */	add	%xg3,512,%xg3



/*    246 */	sxar2
/*    246 */	fmovd	%f470,%f218
/*    246 */	std,s	%f212,[%xg10+448]


/*    246 */	sxar2
/*    246 */	add	%xg9,512,%xg9
/*    246 */	add	%xg1,512,%xg1


/*    246 */	sxar2
/*    246 */	std,s	%f218,[%xg10+464]
/*    246 */	add	%xg10,512,%xg10


/*    246 */	sxar2
/*    246 */	sub	%xg0,8,%xg0
/*    246 */	cmp	%xg0,23

/*    246 */	bge,pt	%icc, .L6974
	nop


.L7107:


/*    246 */	sxar2
/*    246 */	add	%xg11,%xg8,%xg11
/*    246 */	ldd,s	[%xg23],%f38


/*    246 */	fmovd	%f34,%f36




/*    246 */	sxar2
/*    246 */	fmovd	%f32,%f292
/*    246 */	add	%xg12,%xg5,%xg24


/*    246 */	sxar2
/*    246 */	ldd,s	[%xg11],%f46
/*    246 */	fmovd	%f290,%f32



/*    246 */	sxar2
/*    246 */	sra	%xg16,1,%xg16
/*    246 */	add	%xg12,%xg6,%xg25


/*    246 */	sxar2
/*    246 */	ldd,s	[%xg24],%f42
/*    246 */	add	%xg12,%xg7,%xg26


/*    246 */	sxar2
/*    246 */	add	%xg12,%xg8,%xg12
/*    246 */	sra	%xg16,%g0,%xg16


/*    246 */	sxar2
/*    246 */	ldd,s	[%xg25],%f52
/*    246 */	ldd,s	[%xg26],%f48



/*    246 */	sxar2
/*    246 */	mulx	%xg16,416,%xg16
/*    246 */	fmovd	%f294,%f40


/*    246 */	sxar2
/*    246 */	srl	%xg19,31,%xg27
/*    246 */	ldd,s	[%xg12],%f54




/*    246 */	sxar2
/*    246 */	fmovd	%f46,%f294
/*    246 */	fmovd	%f302,%f296



/*    246 */	sxar2
/*    246 */	std,s	%f36,[%xg9]
/*    246 */	add	%xg19,%xg27,%xg19

/*    246 */	sxar1
/*    246 */	add	%xg15,%xg5,%xg28


/*    246 */	fmovd	%f42,%f44



/*    246 */	sxar2
/*    246 */	sra	%xg19,1,%xg19
/*    246 */	add	%xg15,%xg7,%xg30



/*    246 */	sxar2
/*    246 */	fmovd	%f52,%f300
/*    246 */	fmovd	%f298,%f52



/*    246 */	sxar2
/*    246 */	ldd,s	[%xg28],%f56
/*    246 */	sra	%xg19,%g0,%xg19


/*    246 */	fmovd	%f48,%f50



/*    246 */	sxar2
/*    246 */	add	%xg17,%xg20,%xg20
/*    246 */	sra	%xg20,1,%xg20



/*    246 */	sxar2
/*    246 */	fmovd	%f54,%f306
/*    246 */	fmovd	%f304,%f54



/*    246 */	sxar2
/*    246 */	ldd,s	[%xg30],%f74
/*    246 */	add	%xg16,%xg5,%g2


/*    246 */	sxar2
/*    246 */	add	%xg16,%xg7,%g4
/*    246 */	std,s	%f38,[%xg9+16]


/*    246 */	sxar2
/*    246 */	add	%xg16,%xg6,%g3
/*    246 */	add	%xg16,%xg8,%xg16


/*    246 */	sxar2
/*    246 */	add	%xg15,%xg6,%xg29
/*    246 */	std,s	%f32,[%xg10]



/*    246 */	sxar2
/*    246 */	fmovd	%f312,%f58
/*    246 */	mulx	%xg19,416,%xg19


/*    246 */	sxar2
/*    246 */	add	%xg15,%xg8,%xg15
/*    246 */	std,s	%f40,[%xg10+16]



/*    246 */	sxar2
/*    246 */	sra	%xg20,%g0,%xg20
/*    246 */	fmovd	%f330,%f76


/*    246 */	sxar2
/*    246 */	add	%xg17,2,%xg31
/*    246 */	ldd,s	[%g2],%f60


/*    246 */	sxar2
/*    246 */	add	%xg17,4,%g5
/*    246 */	ldd,s	[%g4],%f66


/*    246 */	sxar2
/*    246 */	srl	%xg31,31,%g1
/*    246 */	std,s	%f44,[%xg9+64]

/*    246 */	srl	%g5,31,%o1


/*    246 */	sxar2
/*    246 */	std,s	%f50,[%xg9+80]
/*    246 */	add	%xg31,%g1,%xg31

/*    246 */	add	%g5,%o1,%g5


/*    246 */	sxar2
/*    246 */	ldd,s	[%g3],%f64
/*    246 */	ldd,s	[%xg16],%f70

/*    246 */	sxar1
/*    246 */	sra	%xg31,1,%xg31

/*    246 */	sra	%g5,1,%g5




/*    246 */	sxar2
/*    246 */	fmovd	%f316,%f62
/*    246 */	fmovd	%f322,%f68


/*    246 */	sxar2
/*    246 */	add	%xg19,%xg5,%o2
/*    246 */	add	%xg19,%xg6,%o3


/*    246 */	sxar2
/*    246 */	std,s	%f52,[%xg10+64]
/*    246 */	add	%xg19,%xg7,%o4


/*    246 */	sxar2
/*    246 */	add	%xg19,%xg8,%xg19
/*    246 */	sra	%xg31,%g0,%xg31


/*    246 */	sxar2
/*    246 */	std,s	%f54,[%xg10+80]
/*    246 */	mulx	%xg20,416,%xg20





/*    246 */	sxar2
/*    246 */	fmovd	%f64,%f316
/*    246 */	fmovd	%f70,%f322


/*    246 */	sra	%g5,%g0,%g5


/*    246 */	sxar2
/*    246 */	ldd,s	[%o2],%f80
/*    246 */	add	%xg17,6,%o0


/*    246 */	sxar2
/*    246 */	add	%xg3,512,%xg3
/*    246 */	fmovd	%f320,%f318



/*    246 */	sxar2
/*    246 */	ldd,s	[%o3],%f84
/*    246 */	add	%xg1,512,%xg1


/*    246 */	sxar2
/*    246 */	sub	%xg0,8,%xg0
/*    246 */	ldd,s	[%o4],%f86


/*    246 */	sxar2
/*    246 */	ldd,s	[%xg19],%f90
/*    246 */	ldd,s	[%xg29],%f72



/*    246 */	sxar2
/*    246 */	fmovd	%f68,%f70
/*    246 */	ldd,s	[%xg15],%f78




/*    246 */	sxar2
/*    246 */	fmovd	%f80,%f82
/*    246 */	fmovd	%f84,%f338




/*    246 */	sxar2
/*    246 */	fmovd	%f336,%f84
/*    246 */	add	%xg20,%xg5,%o5


/*    246 */	sxar2
/*    246 */	add	%xg20,%xg6,%o7
/*    246 */	std,s	%f60,[%xg9+128]




/*    246 */	sxar2
/*    246 */	fmovd	%f86,%f88
/*    246 */	fmovd	%f90,%f344



/*    246 */	sxar2
/*    246 */	add	%xg20,%xg7,%xg2
/*    246 */	std,s	%f66,[%xg9+144]




/*    246 */	sxar2
/*    246 */	fmovd	%f72,%f312
/*    246 */	fmovd	%f342,%f90




/*    246 */	sxar2
/*    246 */	add	%xg20,%xg8,%xg20
/*    246 */	fmovd	%f78,%f330




/*    246 */	sxar2
/*    246 */	fmovd	%f58,%f72
/*    246 */	std,s	%f62,[%xg10+128]



/*    246 */	sxar2
/*    246 */	fmovd	%f334,%f332
/*    246 */	std,s	%f70,[%xg10+144]


/*    246 */	sxar2
/*    246 */	mulx	%xg31,416,%xg31
/*    246 */	ldd,s	[%o5],%f92


/*    246 */	sxar2
/*    246 */	ldd,s	[%o7],%f96
/*    246 */	std,s	%f82,[%xg9+192]


/*    246 */	sxar2
/*    246 */	std,s	%f88,[%xg9+208]
/*    246 */	std,s	%f84,[%xg10+192]



/*    246 */	sxar2
/*    246 */	std,s	%f90,[%xg10+208]
/*    246 */	fmovd	%f348,%f94




/*    246 */	sxar2
/*    246 */	fmovd	%f96,%f348
/*    246 */	std,s	%f56,[%xg9+256]



/*    246 */	sxar2
/*    246 */	fmovd	%f352,%f350
/*    246 */	std,s	%f74,[%xg9+272]


/*    246 */	sxar2
/*    246 */	add	%xg31,%xg5,%xg4
/*    246 */	add	%xg31,%xg7,%xg12


/*    246 */	sxar2
/*    246 */	ldd,s	[%xg2],%f98
/*    246 */	ldd,s	[%xg20],%f102


/*    246 */	sxar2
/*    246 */	add	%xg31,%xg6,%xg11
/*    246 */	add	%xg31,%xg8,%xg31

/*    246 */	sxar1
/*    246 */	std,s	%f72,[%xg10+256]

/*    246 */	mulx	%g5,416,%g5


/*    246 */	sxar2
/*    246 */	std,s	%f76,[%xg10+272]
/*    246 */	ldd,s	[%xg4],%f104



/*    246 */	sxar2
/*    246 */	ldd,s	[%xg12],%f110
/*    246 */	fmovd	%f98,%f100



/*    246 */	sxar2
/*    246 */	fmovd	%f102,%f356
/*    246 */	ldd,s	[%xg11],%f108




/*    246 */	sxar2
/*    246 */	fmovd	%f354,%f102
/*    246 */	ldd,s	[%xg31],%f114



/*    246 */	sxar2
/*    246 */	std,s	%f92,[%xg9+320]
/*    246 */	fmovd	%f104,%f106



/*    246 */	sxar2
/*    246 */	fmovd	%f366,%f112
/*    246 */	add	%g5,%xg5,%xg13



/*    246 */	sxar2
/*    246 */	add	%g5,%xg6,%xg14
/*    246 */	fmovd	%f108,%f362




/*    246 */	sxar2
/*    246 */	fmovd	%f360,%f108
/*    246 */	add	%g5,%xg7,%xg15



/*    246 */	sxar2
/*    246 */	add	%g5,%xg8,%g5
/*    246 */	fmovd	%f370,%f368


/*    246 */	sxar2
/*    246 */	fmovd	%f114,%f366
/*    246 */	ldd,s	[%xg13],%f120




/*    246 */	sxar2
/*    246 */	std,s	%f100,[%xg9+336]
/*    246 */	std,s	%f94,[%xg10+320]


/*    246 */	sxar2
/*    246 */	std,s	%f102,[%xg10+336]
/*    246 */	ldd,s	[%xg14],%f124



/*    246 */	sxar2
/*    246 */	ldd,s	[%xg15],%f116
/*    246 */	fmovd	%f120,%f122


/*    246 */	sxar2
/*    246 */	std,s	%f106,[%xg9+384]
/*    246 */	std,s	%f110,[%xg9+400]




/*    246 */	sxar2
/*    246 */	ldd,s	[%g5],%f126
/*    246 */	fmovd	%f116,%f118



/*    246 */	sxar2
/*    246 */	fmovd	%f124,%f378
/*    246 */	std,s	%f108,[%xg10+384]



/*    246 */	sxar2
/*    246 */	fmovd	%f376,%f124
/*    246 */	std,s	%f112,[%xg10+400]




/*    246 */	sxar2
/*    246 */	fmovd	%f126,%f374
/*    246 */	fmovd	%f372,%f126



/*    246 */	sxar2
/*    246 */	std,s	%f122,[%xg9+448]
/*    246 */	std,s	%f118,[%xg9+464]


/*    246 */	sxar2
/*    246 */	add	%xg9,512,%xg9
/*    246 */	std,s	%f124,[%xg10+448]


/*    246 */	sxar2
/*    246 */	std,s	%f126,[%xg10+464]
/*    246 */	add	%xg10,512,%xg10

.L7103:


.L7102:


.L7105:


/*    255 */	sxar2
/*    255 */	srl	%o0,31,%xg24
/*    255 */	add	%o0,2,%xg25


/*    247 */	sxar2
/*    247 */	add	%xg24,%o0,%xg24
/*    247 */	srl	%xg25,31,%xg26


/*    247 */	sxar2
/*    247 */	sra	%xg24,1,%xg24
/*    247 */	add	%xg25,%xg26,%xg26


/*    247 */	sxar2
/*    247 */	sra	%xg24,%g0,%xg24
/*    247 */	sra	%xg26,1,%xg26


/*    247 */	sxar2
/*    247 */	mulx	%xg24,416,%xg24
/*    247 */	sra	%xg26,%g0,%xg26


/*    255 */	sxar2
/*    255 */	add	%xg25,2,%xg25
/*    255 */	add	%xg3,256,%xg3


/*    255 */	sxar2
/*    255 */	srl	%xg25,31,%g2
/*    255 */	add	%xg25,2,%o0

/*    247 */	sxar1
/*    247 */	add	%xg25,%g2,%xg25

/*    247 */	srl	%o0,31,%g3

/*    247 */	sxar1
/*    247 */	sra	%xg25,1,%xg25

/*    247 */	add	%o0,%g3,%g3

/*    247 */	sxar1
/*    247 */	sra	%xg25,%g0,%xg25

/*    247 */	sra	%g3,1,%g3


/*    248 */	sxar2
/*    248 */	add	%xg24,%xg5,%xg27
/*    248 */	add	%xg24,%xg6,%xg28


/*    250 */	sxar2
/*    250 */	add	%xg24,%xg7,%xg29
/*    250 */	add	%xg24,%xg8,%xg24


/*    102 */	sxar2
/*    102 */	ldd,s	[%xg27],%f228
/*    102 */	ldd,s	[%xg28],%f222

/*    102 */	sxar1
/*    102 */	ldd,s	[%xg29],%f224

/*    247 */	sra	%g3,%g0,%g3



/*    247 */	sxar2
/*    247 */	ldd,s	[%xg24],%f230
/*    247 */	mulx	%xg26,416,%xg26


/*    255 */	sxar2
/*    255 */	add	%xg1,256,%xg1
/*    255 */	subcc	%xg0,4,%xg0




/*    102 */	sxar2
/*    102 */	fmovd	%f228,%f220
/*    102 */	fmovd	%f222,%f476





/*    105 */	sxar2
/*    105 */	fmovd	%f224,%f226
/*    105 */	fmovd	%f484,%f222




/*    105 */	sxar2
/*    105 */	fmovd	%f230,%f482
/*    105 */	fmovd	%f480,%f230



/*    248 */	sxar2
/*    248 */	add	%xg26,%xg5,%xg30
/*    248 */	add	%xg26,%xg6,%xg31


/*    250 */	sxar2
/*    250 */	add	%xg26,%xg7,%g1
/*    250 */	add	%xg26,%xg8,%xg26


/*    102 */	sxar2
/*    102 */	ldd,s	[%xg31],%f236
/*    102 */	ldd,s	[%g1],%f242


/*     24 */	sxar2
/*     24 */	ldd,s	[%xg26],%f240
/*     24 */	std,s	%f220,[%xg9]


/*     25 */	sxar2
/*     25 */	mulx	%xg25,416,%xg25
/*     25 */	std,s	%f226,[%xg9+16]



/*    105 */	sxar2
/*    105 */	std,s	%f222,[%xg10]
/*    105 */	fmovd	%f492,%f490




/*    102 */	sxar2
/*    102 */	std,s	%f230,[%xg10+16]
/*    102 */	fmovd	%f240,%f494



/*    102 */	sxar2
/*    102 */	fmovd	%f242,%f238
/*    102 */	ldd,s	[%xg30],%f232



/*    247 */	sxar2
/*    247 */	fmovd	%f498,%f240
/*    247 */	add	%xg25,%xg5,%g4


/*    248 */	sxar2
/*    248 */	add	%xg25,%xg7,%o1
/*    248 */	add	%xg25,%xg6,%g5


/*    102 */	sxar2
/*    102 */	add	%xg25,%xg8,%xg25
/*    102 */	ldd,s	[%g4],%f252



/*    102 */	sxar2
/*    102 */	ldd,s	[%o1],%f248
/*    102 */	fmovd	%f488,%f234




/*    102 */	sxar2
/*    102 */	fmovd	%f236,%f488
/*    102 */	ldd,s	[%g5],%f246

/*     25 */	sxar1
/*     25 */	std,s	%f238,[%xg9+80]

/*    247 */	mulx	%g3,416,%g3




/*    102 */	sxar2
/*    102 */	ldd,s	[%xg25],%f254
/*    102 */	fmovd	%f248,%f250



/*    102 */	sxar2
/*    102 */	fmovd	%f252,%f244
/*    102 */	fmovd	%f246,%f500




/*     24 */	sxar2
/*     24 */	fmovd	%f508,%f246
/*     24 */	std,s	%f232,[%xg9+64]




/*    105 */	sxar2
/*    105 */	fmovd	%f254,%f506
/*    105 */	fmovd	%f504,%f254



/*    248 */	sxar2
/*    248 */	add	%g3,%xg5,%o2
/*    248 */	add	%g3,%xg6,%o3


/*    249 */	sxar2
/*    249 */	std,s	%f234,[%xg10+64]
/*    249 */	add	%g3,%xg7,%o4


/*     25 */	sxar2
/*     25 */	add	%g3,%xg8,%g3
/*     25 */	std,s	%f240,[%xg10+80]


/*    102 */	sxar2
/*    102 */	std,s	%f244,[%xg9+128]
/*    102 */	ldd,s	[%o2],%f32


/*    102 */	sxar2
/*    102 */	ldd,s	[%o3],%f36
/*    102 */	ldd,s	[%o4],%f38


/*     25 */	sxar2
/*     25 */	ldd,s	[%g3],%f42
/*     25 */	std,s	%f250,[%xg9+144]

/*     24 */	sxar1
/*     24 */	std,s	%f246,[%xg10+128]


/*    102 */	fmovd	%f32,%f34


/*    102 */	sxar1
/*    102 */	fmovd	%f36,%f290



/*    102 */	fmovd	%f38,%f40




/*     25 */	sxar2
/*     25 */	fmovd	%f42,%f296
/*     25 */	std,s	%f254,[%xg10+144]



/*    105 */	sxar2
/*    105 */	fmovd	%f288,%f36
/*    105 */	fmovd	%f294,%f42



/*     25 */	sxar2
/*     25 */	std,s	%f34,[%xg9+192]
/*     25 */	std,s	%f40,[%xg9+208]


/*     24 */	sxar2
/*     24 */	add	%xg9,256,%xg9
/*     24 */	std,s	%f36,[%xg10+192]


/*    255 */	sxar2
/*    255 */	std,s	%f42,[%xg10+208]
/*    255 */	add	%xg10,256,%xg10

/*    255 */	bpos,pt	%icc, .L7105
/*    255 */	add	%o0,2,%o0


.L7101:


.L6980:

/*    246 */	sxar1
/*    246 */	addcc	%xg0,3,%xg0

/*    246 */	bneg	.L6975
	nop


.L6981:


/*    246 */	sxar2
/*    246 */	ldx	[%i0+2191],%xg16
/*    246 */	ldx	[%i0+2199],%xg2


/*    246 */	sxar2
/*    246 */	add	%xg16,16,%xg17
/*    246 */	add	%xg16,32,%xg18

/*    246 */	sxar1
/*    246 */	add	%xg16,48,%xg19

.L6988:

/*    247 */	srl	%o0,31,%o5

/*    253 */	sxar1
/*    253 */	add	%xg2,%xg3,%o7

/*    247 */	add	%o5,%o0,%o5

/*    254 */	sxar1
/*    254 */	add	%xg2,%xg1,%xg4

/*    247 */	sra	%o5,1,%o5


/*    247 */	sra	%o5,%g0,%o5

/*    255 */	sxar1
/*    255 */	add	%xg1,64,%xg1

/*    247 */	mulx	%o5,416,%o5


/*    255 */	sxar2
/*    255 */	add	%xg3,64,%xg3
/*    255 */	subcc	%xg0,1,%xg0


/*    248 */	sxar2
/*    248 */	add	%o5,%xg16,%xg5
/*    248 */	add	%o5,%xg17,%xg6


/*    250 */	sxar2
/*    250 */	add	%o5,%xg18,%xg7
/*    250 */	add	%o5,%xg19,%o5


/*    102 */	sxar2
/*    102 */	ldd,s	[%xg5],%f44
/*    102 */	ldd,s	[%xg6],%f48


/*    102 */	sxar2
/*    102 */	ldd,s	[%xg7],%f50
/*    102 */	ldd,s	[%o5],%f54


/*    102 */	fmovd	%f44,%f46


/*    102 */	sxar1
/*    102 */	fmovd	%f48,%f302


/*    102 */	fmovd	%f50,%f52




/*    105 */	sxar2
/*    105 */	fmovd	%f54,%f308
/*    105 */	fmovd	%f300,%f48





/*     24 */	sxar2
/*     24 */	fmovd	%f306,%f54
/*     24 */	std,s	%f46,[%o7]


/*     24 */	sxar2
/*     24 */	std,s	%f52,[%o7+16]
/*     24 */	std,s	%f48,[%xg4]

/*     25 */	sxar1
/*     25 */	std,s	%f54,[%xg4+16]

/*    255 */	bpos,pt	%icc, .L6988
/*    255 */	add	%o0,2,%o0


.L6982:


.L6975:

/*    255 */
/*    255 */	ba	.L6972
	nop


.L6977:

/*    255 *//*    255 */	call	__mpc_obar
/*    255 */	ldx	[%fp+2199],%o0

/*    255 *//*    255 */	call	__mpc_obar
/*    255 */	ldx	[%fp+2199],%o0


.L6978:

/*    255 */	ret
	restore



.LLFE10:
	.size	_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_4,.-_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_4
	.type	_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_4,#function
	.ident	"$Compiler: Fujitsu C/C++ Compiler Version 1.2.1 P-id: T01641-01 (Jun  7 2013 14:39:28) ../SRC/hermite8-k.cpp _ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_5 $"
	.section	".text"
	.align	64
_ZN7Gravity17calc_potential_rpEidPKNS_9GParticleEP4v4r8Pd._OMP_5:
.LLFB11:
.L6990:

/*    257 */	save	%sp,-2048,%sp
.LLCFI9:
/*    257 */	stx	%i0,[%fp+2175]
/*    257 */	stx	%i3,[%fp+2199]
/*    257 */	stx	%i0,[%fp+2175]

.L6991:

/*    257 *//*    257 */	ldsw	[%i0+2035],%g1
/*    257 */
/*    257 */
/*    257 */
/*    258 */	ldsw	[%i0+2179],%l0
/*    258 */	cmp	%l0,%g0
/*    258 */	ble	.L7005
/*    258 */	mov	%g0,%o0


.L6992:

/*    258 */	sxar1
/*    258 */	fzero,s	%f34

/*    258 */	sethi	%h44(.LR0.cnt.6),%g1

/*    258 */	sxar1
/*    258 */	sethi	%h44(.LR0.cnt.7),%xg0

/*    258 */	or	%g1,%m44(.LR0.cnt.6),%g1

/*    258 */	sxar1
/*    258 */	or	%xg0,%m44(.LR0.cnt.7),%xg0

/*    258 */	sllx	%g1,12,%g1

/*    258 */	sxar1
/*    258 */	sllx	%xg0,12,%xg0

/*    258 */	or	%g1,%l44(.LR0.cnt.6),%g1


/*    258 */	sxar2
/*    258 */	or	%xg0,%l44(.LR0.cnt.7),%xg0
/*    258 */	mov	1,%xg31

/*    258 */	sra	%l0,%g0,%l0


/*    258 */	sxar2
/*    258 */	ldd	[%g1],%f232
/*    258 */	ldd	[%g1],%f488



/*    258 */	sxar2
/*    258 */	ldd	[%xg0],%f234
/*    258 */	ldd	[%xg0],%f490



/*    258 */	sxar2
/*    ??? */	std,s	%f34,[%fp+223]
/*    258 */	stx	%xg31,[%fp+2031]


/*    258 */	sxar2
/*    ??? */	std,s	%f232,[%fp+255]
/*    ??? */	std,s	%f234,[%fp+239]

.L7022:

/*    258 */	add	%fp,2039,%l1

/*    258 */	mov	1,%l5

/*    258 */	add	%fp,2023,%l2

/*    258 */	add	%fp,2031,%l3

/*    258 */	sra	%l5,%g0,%l4

.L6994:

/*    258 */	sra	%o0,%g0,%o0

/*    258 */	stx	%g0,[%sp+2223]

/*    258 */	mov	4,%o2

/*    258 */	mov	%g0,%o3

/*    258 */	mov	%l0,%o1

/*    258 */	mov	%l1,%o4


/*    258 */	stx	%g0,[%sp+2231]

/*    258 */	stx	%l3,[%sp+2239]


/*    258 */	sxar2
/*    258 */	ldx	[%fp+2199],%xg29
/*    258 */	stx	%xg29,[%sp+2247]

/*    258 */	call	__mpc_ostd_th
/*    258 */	mov	%l2,%o5
/*    258 */	sxar2
/*    258 */	ldx	[%fp+2031],%xg30
/*    258 */	cmp	%xg30,%g0
/*    258 */	ble,pn	%xcc, .L7005
	nop


.L6995:

/*    258 */	ldx	[%fp+2039],%o0


/*    258 */	sxar2
/*    258 */	ldx	[%fp+2023],%xg0
/*    258 */	ldd	[%i0+2183],%f74



/*    258 */	sxar2
/*    258 */	ldd	[%i0+2183],%f330
/*    258 */	ldsw	[%i0+2179],%xg9


/*    258 */	sxar2
/*    258 */	ldx	[%i0+2199],%xg5
/*    258 */	ldx	[%i0+2207],%xg19

/*    258 */	sra	%o0,%g0,%o0


/*    258 */	sxar2
/*    258 */	sra	%xg0,%g0,%xg0
/*    258 */	sub	%xg0,%o0,%xg0


/*    258 */	sxar2
/*    258 */	sra	%o0,%g0,%xg1
/*    258 */	sra	%xg0,1,%xg2


/*    258 */	sxar2
/*    258 */	sllx	%xg1,5,%xg3
/*    258 */	srl	%xg2,30,%xg2


/*    258 */	sxar2
/*    258 */	sllx	%xg1,3,%xg1
/*    258 */	add	%xg0,%xg2,%xg0


/*    258 */	sxar2
/*    258 */	add	%xg5,32,%xg4
/*    258 */	sra	%xg0,2,%xg0


/*    258 */	sxar2
/*    258 */	add	%xg0,1,%xg0
/*    258 */	sra	%xg0,%g0,%xg0


/*    258 */	sxar2
/*    258 */	sub	%l4,%xg0,%xg0
/*    258 */	srax	%xg0,32,%xg6


/*    258 */	sxar2
/*    258 */	and	%xg0,%xg6,%xg0
/*    258 */	sub	%l5,%xg0,%xg0

.L6996:


/*    265 */	sxar2
/*    265 */	add	%xg5,%xg3,%xg7
/*    265 */	cmp	%xg9,%g0


/*    260 */	sxar2
/*    ??? */	ldd,s	[%fp+223],%f32
/*    260 */	ldd	[%xg7],%f250



/*    260 */	sxar2
/*    260 */	ldd	[%xg7+32],%f506
/*    260 */	ldd	[%xg7+64],%f38



/*     37 */	sxar2
/*     37 */	ldd	[%xg7+96],%f294
/*     37 */	std,s	%f250,[%fp+1135]


/*    261 */	sxar2
/*    261 */	std,s	%f38,[%fp+1151]
/*    261 */	ldd	[%xg7+8],%f252



/*    261 */	sxar2
/*    261 */	ldd	[%xg7+40],%f508
/*    261 */	ldd	[%xg7+72],%f46



/*     37 */	sxar2
/*     37 */	ldd	[%xg7+104],%f302
/*     37 */	std,s	%f252,[%fp+1167]


/*    262 */	sxar2
/*    262 */	std,s	%f46,[%fp+1183]
/*    262 */	ldd	[%xg7+16],%f254



/*    262 */	sxar2
/*    262 */	ldd	[%xg7+48],%f510
/*    262 */	ldd	[%xg7+80],%f56



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

/*    265 */	ble	.L7002
	nop


.L6998:


/*    277 */	sxar2
/*    277 */	ldd,s	[%fp+1135],%f32
/*    277 */	mov	%g0,%xg10


/*    277 */	sxar2
/*    ??? */	ldd,s	[%fp+223],%f186
/*    277 */	subcc	%xg9,2,%xg8


/*    265 */	sxar2
/*    265 */	ldd,s	[%fp+1263],%f184
/*    265 */	ldd,s	[%fp+1167],%f42


/*    265 */	sxar2
/*    265 */	ldd,s	[%fp+1199],%f50
/*    265 */	ldd,s	[%fp+1231],%f72

/*    277 */	bneg	.L7008
	nop


.L7011:


/*    265 */	sxar2
/*    265 */	ldx	[%i0+2199],%xg12
/*    265 */	cmp	%xg8,14

/*    265 */	bl	.L7112
	nop


.L7108:


.L7115:


/*    265 */	sxar2
/*    265 */	add	%xg12,%xg10,%xg11
/*    265 */	add	%xg4,%xg10,%xg13


/*    265 */	sxar2
/*    ??? */	ldd,s	[%fp+255],%f158
/*    265 */	ldd,s	[%xg11],%f36


/*    265 */	sxar2
/*    265 */	ldd,s	[%xg11+16],%f52
/*    265 */	add	%xg10,64,%xg14


/*    265 */	sxar2
/*    265 */	add	%xg10,128,%xg10
/*    265 */	add	%xg12,%xg14,%xg15


/*    265 */	sxar2
/*    265 */	ldd,s	[%xg13],%f60
/*    265 */	ldd,s	[%xg13+16],%f68


/*    265 */	sxar2
/*    265 */	add	%xg4,%xg14,%xg14
/*    265 */	ldd,s	[%xg15],%f78


/*    265 */	sxar2
/*    265 */	add	%xg12,%xg10,%xg16
/*    ??? */	ldd,s	[%fp+239],%f160


/*    265 */	sxar2
/*    265 */	ldd,s	[%xg15+16],%f86
/*    265 */	ldd,s	[%xg14],%f92


/*    265 */	sxar2
/*    265 */	ldd,s	[%xg16],%f100
/*    265 */	fnmsubd,sc	%f36,%f158,%f32,%f34


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f36,%f158,%f38,%f40
/*    265 */	ldd,s	[%xg14+16],%f106


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f292,%f158,%f42,%f44
/*    265 */	fnmsubd,sc	%f292,%f158,%f46,%f36


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f52,%f158,%f50,%f48
/*    265 */	fnmsubd,sc	%f52,%f158,%f56,%f54


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f60,%f158,%f32,%f58
/*    265 */	fnmsubd,sc	%f60,%f158,%f38,%f62


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f316,%f158,%f42,%f64
/*    265 */	fnmsubd,sc	%f316,%f158,%f46,%f60


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f68,%f158,%f50,%f66
/*    265 */	fnmsubd,sc	%f68,%f158,%f56,%f70


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f34,%f34,%f72,%f34
/*    265 */	fmaddd,s	%f40,%f40,%f74,%f40


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f78,%f158,%f32,%f76
/*    265 */	fnmsubd,sc	%f78,%f158,%f38,%f80


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f334,%f158,%f42,%f82
/*    265 */	fnmsubd,sc	%f334,%f158,%f46,%f78


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f58,%f58,%f72,%f58
/*    265 */	fmaddd,s	%f62,%f62,%f74,%f62


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f86,%f158,%f50,%f84
/*    265 */	fnmsubd,sc	%f86,%f158,%f56,%f88


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f92,%f158,%f32,%f90
/*    265 */	fnmsubd,sc	%f92,%f158,%f38,%f94


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f44,%f44,%f34,%f44
/*    265 */	fmaddd,s	%f36,%f36,%f40,%f36


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f76,%f76,%f72,%f76
/*    265 */	fmaddd,s	%f80,%f80,%f74,%f80


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f348,%f158,%f42,%f96
/*    265 */	fnmsubd,sc	%f348,%f158,%f46,%f92


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f64,%f64,%f58,%f64
/*    265 */	fmaddd,s	%f60,%f60,%f62,%f60


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f100,%f158,%f32,%f98
/*    265 */	fnmsubd,sc	%f100,%f158,%f38,%f102


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f90,%f90,%f72,%f90
/*    265 */	fmaddd,s	%f94,%f94,%f74,%f94


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f48,%f48,%f44,%f48
/*    265 */	fmaddd,s	%f54,%f54,%f36,%f54


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f82,%f82,%f76,%f82
/*    265 */	fmaddd,s	%f78,%f78,%f80,%f78


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f106,%f158,%f50,%f104
/*    265 */	fnmsubd,sc	%f106,%f158,%f56,%f108


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f66,%f66,%f64,%f66
/*    265 */	fmaddd,s	%f70,%f70,%f60,%f70


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f96,%f96,%f90,%f96
/*    265 */	fmaddd,s	%f92,%f92,%f94,%f92


/*    265 */	sxar2
/*    265 */	frsqrtad,s	%f48,%f110
/*    265 */	frsqrtad,s	%f54,%f112


/*    265 */	sxar2
/*    265 */	fmuld,s	%f48,%f160,%f114
/*    265 */	fmuld,s	%f54,%f160,%f116


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f84,%f84,%f82,%f84
/*    265 */	fmaddd,s	%f88,%f88,%f78,%f88


/*    265 */	sxar2
/*    265 */	frsqrtad,s	%f66,%f118
/*    265 */	frsqrtad,s	%f70,%f120


/*    265 */	sxar2
/*    265 */	fmuld,s	%f66,%f160,%f122
/*    265 */	fmuld,s	%f70,%f160,%f124


/*    265 */	sxar2
/*    265 */	fmuld,s	%f110,%f110,%f126
/*    265 */	fmuld,s	%f112,%f112,%f128


/*    265 */	sxar2
/*    265 */	frsqrtad,s	%f84,%f130
/*    265 */	frsqrtad,s	%f88,%f132


/*    265 */	sxar2
/*    265 */	fmuld,s	%f118,%f118,%f134
/*    265 */	fmuld,s	%f120,%f120,%f136


/*    265 */	sxar2
/*    265 */	fmuld,s	%f84,%f160,%f138
/*    265 */	fmuld,s	%f88,%f160,%f140


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f114,%f126,%f160,%f126
/*    265 */	fnmsubd,s	%f116,%f128,%f160,%f128


/*    265 */	sxar2
/*    265 */	fmuld,s	%f130,%f130,%f142
/*    265 */	fmuld,s	%f132,%f132,%f144


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f122,%f134,%f160,%f134
/*    265 */	fnmsubd,s	%f124,%f136,%f160,%f136


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f110,%f126,%f110,%f110
/*    265 */	fmaddd,s	%f112,%f128,%f112,%f112


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f118,%f134,%f118,%f118
/*    265 */	fmaddd,s	%f120,%f136,%f120,%f120


/*    265 */	sxar2
/*    265 */	fmuld,s	%f110,%f110,%f146
/*    265 */	fmuld,s	%f112,%f112,%f148


/*    265 */	sxar2
/*    265 */	fmuld,s	%f118,%f118,%f150
/*    265 */	fmuld,s	%f120,%f120,%f152


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f114,%f146,%f160,%f146
/*    265 */	fnmsubd,s	%f116,%f148,%f160,%f148


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f122,%f150,%f160,%f150
/*    265 */	fnmsubd,s	%f124,%f152,%f160,%f152


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f110,%f146,%f110,%f110
/*    265 */	fmaddd,s	%f112,%f148,%f112,%f112


/*    265 */	sxar2
/*    265 */	fmuld,s	%f110,%f110,%f154
/*    265 */	fmuld,s	%f112,%f112,%f156

.L7000:


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f356,%f158,%f42,%f36
/*    265 */	fnmsubd,sc	%f356,%f158,%f46,%f100


/*    265 */	sxar2
/*    265 */	ldd,s	[%xg16+16],%f164
/*    265 */	fmaddd,s	%f98,%f98,%f72,%f98


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f104,%f104,%f96,%f104
/*    265 */	fmaddd,s	%f102,%f102,%f74,%f102


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f108,%f108,%f92,%f108
/*    265 */	fmaddd,s	%f118,%f150,%f118,%f118


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f120,%f152,%f120,%f120
/*    265 */	fnmsubd,s	%f138,%f142,%f160,%f142


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f114,%f154,%f160,%f114
/*    265 */	fnmsubd,s	%f140,%f144,%f160,%f144


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f116,%f156,%f160,%f116
/*    265 */	fnmsubd,sc	%f164,%f158,%f50,%f162


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f164,%f158,%f56,%f166
/*    265 */	fmaddd,s	%f36,%f36,%f98,%f36


/*    265 */	sxar2
/*    265 */	frsqrtad,s	%f104,%f168
/*    265 */	fmaddd,s	%f100,%f100,%f102,%f100


/*    265 */	sxar2
/*    265 */	frsqrtad,s	%f108,%f170
/*    265 */	add	%xg10,64,%xg20


/*    265 */	sxar2
/*    265 */	fmuld,s	%f118,%f118,%f172
/*    265 */	fmuld,s	%f120,%f120,%f174


/*    265 */	sxar2
/*    265 */	add	%xg12,%xg20,%xg21
/*    265 */	fmaddd,s	%f130,%f142,%f130,%f130


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f110,%f114,%f110,%f110
/*    265 */	ldd,s	[%xg21],%f190


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f132,%f144,%f132,%f132
/*    265 */	fmaddd,s	%f112,%f116,%f112,%f112


/*    265 */	sxar2
/*    265 */	fcmplted,s	%f72,%f48,%f48
/*    265 */	fcmplted,s	%f74,%f54,%f54


/*    265 */	sxar2
/*    265 */	fmuld,s	%f104,%f160,%f176
/*    265 */	fmuld,s	%f168,%f168,%f178


/*    265 */	sxar2
/*    265 */	fmuld,s	%f108,%f160,%f180
/*    265 */	fmuld,s	%f170,%f170,%f182


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f122,%f172,%f160,%f122
/*    265 */	fnmsubd,s	%f124,%f174,%f160,%f124


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f190,%f158,%f32,%f188
/*    265 */	fnmsubd,sc	%f190,%f158,%f38,%f192


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f162,%f162,%f36,%f162
/*    265 */	fmuld,s	%f130,%f130,%f194


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f166,%f166,%f100,%f166
/*    265 */	fmuld,s	%f132,%f132,%f196


/*    265 */	sxar2
/*    265 */	add	%xg4,%xg10,%xg22
/*    265 */	fand,s	%f110,%f48,%f110


/*    265 */	sxar2
/*    265 */	fand,s	%f112,%f54,%f112
/*    265 */	ldd,s	[%xg22],%f204


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f176,%f178,%f160,%f178
/*    265 */	fnmsubd,s	%f180,%f182,%f160,%f182


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f118,%f122,%f118,%f118
/*    265 */	fmaddd,s	%f120,%f124,%f120,%f120


/*    265 */	sxar2
/*    265 */	fcmplted,s	%f72,%f66,%f66
/*    265 */	fcmplted,s	%f74,%f70,%f70


/*    265 */	sxar2
/*    265 */	frsqrtad,s	%f162,%f198
/*    265 */	fnmsubd,s	%f138,%f194,%f160,%f194


/*    265 */	sxar2
/*    265 */	fmuld,s	%f162,%f160,%f200
/*    265 */	fnmsubd,sc	%f204,%f158,%f32,%f202


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f140,%f196,%f160,%f196
/*    265 */	fnmsubd,sc	%f204,%f158,%f38,%f206


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f168,%f178,%f168,%f168
/*    265 */	fmaddd,s	%f170,%f182,%f170,%f170


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f308,%f110,%f184,%f110
/*    265 */	fnmsubd,sc	%f308,%f112,%f186,%f52


/*    265 */	sxar2
/*    265 */	fand,s	%f118,%f66,%f118
/*    265 */	fand,s	%f120,%f70,%f120


/*    265 */	sxar2
/*    265 */	fmuld,s	%f198,%f198,%f208
/*    265 */	fmaddd,s	%f130,%f194,%f130,%f130


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f460,%f158,%f42,%f210
/*    265 */	fmaddd,s	%f202,%f202,%f72,%f202


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f460,%f158,%f46,%f204
/*    265 */	ldd,s	[%xg22+16],%f186


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f206,%f206,%f74,%f206
/*    265 */	fmuld,s	%f168,%f168,%f212


/*    265 */	sxar2
/*    265 */	fmuld,s	%f170,%f170,%f214
/*    265 */	frsqrtad,s	%f166,%f216


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f132,%f196,%f132,%f132
/*    265 */	fnmsubd,sc	%f324,%f118,%f110,%f118


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f324,%f120,%f52,%f68
/*    265 */	fmuld,s	%f130,%f130,%f218


/*    265 */	sxar2
/*    265 */	fmuld,s	%f166,%f160,%f220
/*    265 */	fnmsubd,sc	%f186,%f158,%f50,%f222


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f210,%f210,%f202,%f210
/*    265 */	fnmsubd,sc	%f186,%f158,%f56,%f224


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f204,%f204,%f206,%f204
/*    265 */	fnmsubd,s	%f176,%f212,%f160,%f212


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f180,%f214,%f160,%f214
/*    265 */	fmuld,s	%f216,%f216,%f226


/*    265 */	sxar2
/*    265 */	fmuld,s	%f132,%f132,%f228
/*    265 */	fnmsubd,sc	%f446,%f158,%f42,%f230


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f446,%f158,%f46,%f190
/*    265 */	ldd,s	[%xg21+16],%f52


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f188,%f188,%f72,%f188
/*    265 */	fmaddd,s	%f222,%f222,%f210,%f222


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f192,%f192,%f74,%f192
/*    265 */	fmaddd,s	%f224,%f224,%f204,%f224


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f168,%f212,%f168,%f168
/*    265 */	fmaddd,s	%f170,%f214,%f170,%f170


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f200,%f208,%f160,%f208
/*    265 */	fnmsubd,s	%f138,%f218,%f160,%f138


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f220,%f226,%f160,%f226
/*    265 */	fnmsubd,s	%f140,%f228,%f160,%f140


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f52,%f158,%f50,%f48
/*    265 */	fnmsubd,sc	%f52,%f158,%f56,%f54


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f230,%f230,%f188,%f230
/*    265 */	frsqrtad,s	%f222,%f184


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f190,%f190,%f192,%f190
/*    265 */	frsqrtad,s	%f224,%f232


/*    265 */	sxar2
/*    265 */	add	%xg10,128,%xg23
/*    265 */	fmuld,s	%f168,%f168,%f234


/*    265 */	sxar2
/*    265 */	fmuld,s	%f170,%f170,%f236
/*    265 */	add	%xg12,%xg23,%xg24


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f198,%f208,%f198,%f198
/*    265 */	fmaddd,s	%f130,%f138,%f130,%f130


/*    265 */	sxar2
/*    265 */	ldd,s	[%xg24],%f248
/*    265 */	fmaddd,s	%f216,%f226,%f216,%f216


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f132,%f140,%f132,%f132
/*    265 */	fcmplted,s	%f72,%f84,%f84


/*    265 */	sxar2
/*    265 */	fcmplted,s	%f74,%f88,%f88
/*    265 */	fmuld,s	%f222,%f160,%f238


/*    265 */	sxar2
/*    265 */	fmuld,s	%f184,%f184,%f240
/*    265 */	fmuld,s	%f224,%f160,%f242


/*    265 */	sxar2
/*    265 */	fmuld,s	%f232,%f232,%f244
/*    265 */	fnmsubd,s	%f176,%f234,%f160,%f176


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f180,%f236,%f160,%f180
/*    265 */	fnmsubd,sc	%f248,%f158,%f32,%f246


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f248,%f158,%f38,%f250
/*    265 */	fmaddd,s	%f48,%f48,%f230,%f48


/*    265 */	sxar2
/*    265 */	fmuld,s	%f198,%f198,%f252
/*    265 */	fmaddd,s	%f54,%f54,%f190,%f54


/*    265 */	sxar2
/*    265 */	fmuld,s	%f216,%f216,%f254
/*    265 */	add	%xg4,%xg20,%xg20


/*    265 */	sxar2
/*    265 */	fand,s	%f130,%f84,%f130
/*    265 */	fand,s	%f132,%f88,%f132


/*    265 */	sxar2
/*    265 */	ldd,s	[%xg20],%f36
/*    265 */	fnmsubd,s	%f238,%f240,%f160,%f240


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f242,%f244,%f160,%f244
/*    265 */	fmaddd,s	%f168,%f176,%f168,%f168


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f170,%f180,%f170,%f170
/*    265 */	fcmplted,s	%f72,%f104,%f104


/*    265 */	sxar2
/*    265 */	fcmplted,s	%f74,%f108,%f108
/*    265 */	frsqrtad,s	%f48,%f110


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f200,%f252,%f160,%f252
/*    265 */	fmuld,s	%f48,%f160,%f114


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f36,%f158,%f32,%f34
/*    265 */	fnmsubd,s	%f220,%f254,%f160,%f254


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f36,%f158,%f38,%f40
/*    265 */	fmaddd,s	%f184,%f240,%f184,%f184


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f232,%f244,%f232,%f232
/*    265 */	fnmsubd,sc	%f342,%f130,%f118,%f130


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f342,%f132,%f68,%f86
/*    265 */	fand,s	%f168,%f104,%f168


/*    265 */	sxar2
/*    265 */	fand,s	%f170,%f108,%f170
/*    265 */	fmuld,s	%f110,%f110,%f44


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f198,%f252,%f198,%f198
/*    265 */	fnmsubd,sc	%f292,%f158,%f42,%f58


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f34,%f34,%f72,%f34
/*    265 */	fnmsubd,sc	%f292,%f158,%f46,%f36


/*    265 */	sxar2
/*    265 */	ldd,s	[%xg20+16],%f68
/*    265 */	fmaddd,s	%f40,%f40,%f74,%f40


/*    265 */	sxar2
/*    265 */	fmuld,s	%f184,%f184,%f60
/*    265 */	fmuld,s	%f232,%f232,%f62


/*    265 */	sxar2
/*    265 */	frsqrtad,s	%f54,%f112
/*    265 */	fmaddd,s	%f216,%f254,%f216,%f216


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f362,%f168,%f130,%f168
/*    265 */	fnmsubd,sc	%f362,%f170,%f86,%f106


/*    265 */	sxar2
/*    265 */	fmuld,s	%f198,%f198,%f64
/*    265 */	fmuld,s	%f54,%f160,%f116


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f68,%f158,%f50,%f66
/*    265 */	fmaddd,s	%f58,%f58,%f34,%f58


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f68,%f158,%f56,%f70
/*    265 */	fmaddd,s	%f36,%f36,%f40,%f36


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f238,%f60,%f160,%f60
/*    265 */	fnmsubd,s	%f242,%f62,%f160,%f62


/*    265 */	sxar2
/*    265 */	fmuld,s	%f112,%f112,%f76
/*    265 */	fmuld,s	%f216,%f216,%f78


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f504,%f158,%f42,%f80
/*    265 */	fnmsubd,sc	%f504,%f158,%f46,%f248


/*    265 */	sxar2
/*    265 */	ldd,s	[%xg24+16],%f86
/*    265 */	fmaddd,s	%f246,%f246,%f72,%f246


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f66,%f66,%f58,%f66
/*    265 */	fmaddd,s	%f250,%f250,%f74,%f250


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f70,%f70,%f36,%f70
/*    265 */	fmaddd,s	%f184,%f60,%f184,%f184


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f232,%f62,%f232,%f232
/*    265 */	fnmsubd,s	%f114,%f44,%f160,%f44


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f200,%f64,%f160,%f200
/*    265 */	fnmsubd,s	%f116,%f76,%f160,%f76


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f220,%f78,%f160,%f220
/*    265 */	fnmsubd,sc	%f86,%f158,%f50,%f84


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f86,%f158,%f56,%f88
/*    265 */	fmaddd,s	%f80,%f80,%f246,%f80


/*    265 */	sxar2
/*    265 */	frsqrtad,s	%f66,%f118
/*    265 */	fmaddd,s	%f248,%f248,%f250,%f248


/*    265 */	sxar2
/*    265 */	frsqrtad,s	%f70,%f120
/*    265 */	add	%xg10,192,%xg10


/*    265 */	sxar2
/*    265 */	fmuld,s	%f184,%f184,%f82
/*    265 */	fmuld,s	%f232,%f232,%f90


/*    265 */	sxar2
/*    265 */	add	%xg12,%xg10,%xg16
/*    265 */	fmaddd,s	%f110,%f44,%f110,%f110


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f198,%f200,%f198,%f198
/*    265 */	ldd,s	[%xg16],%f100


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f112,%f76,%f112,%f112
/*    265 */	fmaddd,s	%f216,%f220,%f216,%f216


/*    265 */	sxar2
/*    265 */	fcmplted,s	%f72,%f162,%f162
/*    265 */	fcmplted,s	%f74,%f166,%f166


/*    265 */	sxar2
/*    265 */	fmuld,s	%f66,%f160,%f122
/*    265 */	fmuld,s	%f118,%f118,%f94


/*    265 */	sxar2
/*    265 */	fmuld,s	%f70,%f160,%f124
/*    265 */	fmuld,s	%f120,%f120,%f96


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f238,%f82,%f160,%f238
/*    265 */	fnmsubd,s	%f242,%f90,%f160,%f242


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f100,%f158,%f32,%f98
/*    265 */	fnmsubd,sc	%f100,%f158,%f38,%f102


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f84,%f84,%f80,%f84
/*    265 */	fmuld,s	%f110,%f110,%f104


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f88,%f88,%f248,%f88
/*    265 */	fmuld,s	%f112,%f112,%f108


/*    265 */	sxar2
/*    265 */	add	%xg4,%xg23,%xg23
/*    265 */	fand,s	%f198,%f162,%f198


/*    265 */	sxar2
/*    265 */	fand,s	%f216,%f166,%f216
/*    265 */	ldd,s	[%xg23],%f92


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f122,%f94,%f160,%f94
/*    265 */	fnmsubd,s	%f124,%f96,%f160,%f96


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f184,%f238,%f184,%f184
/*    265 */	fmaddd,s	%f232,%f242,%f232,%f232


/*    265 */	sxar2
/*    265 */	fcmplted,s	%f72,%f222,%f222
/*    265 */	fcmplted,s	%f74,%f224,%f224


/*    265 */	sxar2
/*    265 */	frsqrtad,s	%f84,%f130
/*    265 */	fnmsubd,s	%f114,%f104,%f160,%f104


/*    265 */	sxar2
/*    265 */	fmuld,s	%f84,%f160,%f138
/*    265 */	fnmsubd,sc	%f92,%f158,%f32,%f126


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f116,%f108,%f160,%f108
/*    265 */	fnmsubd,sc	%f92,%f158,%f38,%f128


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f118,%f94,%f118,%f118
/*    265 */	fmaddd,s	%f120,%f96,%f120,%f120


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f420,%f198,%f168,%f198
/*    265 */	fnmsubd,sc	%f420,%f216,%f106,%f164


/*    265 */	sxar2
/*    265 */	fand,s	%f184,%f222,%f184
/*    265 */	fand,s	%f232,%f224,%f232


/*    265 */	sxar2
/*    265 */	fmuld,s	%f130,%f130,%f142
/*    265 */	fmaddd,s	%f110,%f104,%f110,%f110


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f348,%f158,%f42,%f96
/*    265 */	fmaddd,s	%f126,%f126,%f72,%f126


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f348,%f158,%f46,%f92
/*    265 */	ldd,s	[%xg23+16],%f106


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f128,%f128,%f74,%f128
/*    265 */	fmuld,s	%f118,%f118,%f150


/*    265 */	sxar2
/*    265 */	fmuld,s	%f120,%f120,%f152
/*    265 */	frsqrtad,s	%f88,%f132


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f112,%f108,%f112,%f112
/*    265 */	fnmsubd,sc	%f442,%f184,%f198,%f184


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f442,%f232,%f164,%f186
/*    265 */	fmuld,s	%f110,%f110,%f154


/*    265 */	sxar2
/*    265 */	fmuld,s	%f88,%f160,%f140
/*    265 */	fnmsubd,sc	%f106,%f158,%f50,%f104


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f96,%f96,%f126,%f96
/*    265 */	fnmsubd,sc	%f106,%f158,%f56,%f108


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f92,%f92,%f128,%f92
/*    265 */	fnmsubd,s	%f122,%f150,%f160,%f150


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f124,%f152,%f160,%f152
/*    265 */	fmuld,s	%f132,%f132,%f144


/*    265 */	sxar2
/*    265 */	fmuld,s	%f112,%f112,%f156
/*    265 */	sub	%xg8,6,%xg8

/*    265 */	sxar1
/*    265 */	cmp	%xg8,15

/*    265 */	bge,pt	%icc, .L7000
	nop


.L7116:


/*    265 */	sxar2
/*    ??? */	ldd,s	[%fp+255],%f248
/*    265 */	ldd,s	[%xg16+16],%f162


/*    265 */	sxar2
/*    265 */	add	%xg4,%xg10,%xg17
/*    265 */	fmaddd,s	%f98,%f98,%f72,%f98


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f104,%f104,%f96,%f104
/*    265 */	ldd,s	[%xg17],%f168


/*    265 */	sxar2
/*    265 */	ldd,s	[%xg17+16],%f188
/*    265 */	fmaddd,s	%f102,%f102,%f74,%f102


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f108,%f108,%f92,%f108
/*    265 */	add	%xg10,64,%xg10


/*    265 */	sxar2
/*    ??? */	ldd,s	[%fp+239],%f236
/*    265 */	fcmplted,s	%f72,%f48,%f48


/*    265 */	sxar2
/*    265 */	sub	%xg8,6,%xg8
/*    265 */	fmaddd,s	%f118,%f150,%f118,%f118


/*    265 */	sxar2
/*    265 */	fcmplted,s	%f74,%f54,%f54
/*    265 */	fnmsubd,sc	%f356,%f248,%f42,%f158


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f356,%f248,%f46,%f100
/*    265 */	fnmsubd,sc	%f162,%f248,%f50,%f160


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f168,%f248,%f32,%f166
/*    265 */	fnmsubd,sc	%f168,%f248,%f38,%f170


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f424,%f248,%f42,%f172
/*    265 */	fnmsubd,sc	%f162,%f248,%f56,%f164


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f424,%f248,%f46,%f168
/*    265 */	fnmsubd,s	%f138,%f142,%f236,%f142


/*    265 */	sxar2
/*    265 */	frsqrtad,s	%f104,%f174
/*    265 */	frsqrtad,s	%f108,%f176


/*    265 */	sxar2
/*    265 */	fmuld,s	%f104,%f236,%f178
/*    265 */	fmaddd,s	%f158,%f158,%f98,%f158


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f100,%f100,%f102,%f100
/*    265 */	fnmsubd,sc	%f188,%f248,%f50,%f182


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f166,%f166,%f72,%f166
/*    265 */	fnmsubd,sc	%f188,%f248,%f56,%f190


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f170,%f170,%f74,%f170
/*    265 */	fnmsubd,s	%f140,%f144,%f236,%f144


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f114,%f154,%f236,%f114
/*    265 */	fmuld,s	%f108,%f236,%f180


/*    265 */	sxar2
/*    265 */	fmuld,s	%f174,%f174,%f192
/*    265 */	fmaddd,s	%f130,%f142,%f130,%f130


/*    265 */	sxar2
/*    265 */	fmuld,s	%f176,%f176,%f194
/*    265 */	fmaddd,s	%f160,%f160,%f158,%f160


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f164,%f164,%f100,%f164
/*    265 */	fnmsubd,s	%f116,%f156,%f236,%f116


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f172,%f172,%f166,%f172
/*    265 */	fmaddd,s	%f120,%f152,%f120,%f120


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f168,%f168,%f170,%f168
/*    265 */	fmaddd,s	%f132,%f144,%f132,%f132


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f110,%f114,%f110,%f110
/*    265 */	fnmsubd,s	%f178,%f192,%f236,%f192


/*    265 */	sxar2
/*    265 */	fmuld,s	%f118,%f118,%f208
/*    265 */	fmuld,s	%f130,%f130,%f196


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f180,%f194,%f236,%f194
/*    265 */	frsqrtad,s	%f160,%f200


/*    265 */	sxar2
/*    265 */	frsqrtad,s	%f164,%f202
/*    265 */	fmuld,s	%f160,%f236,%f204


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f182,%f182,%f172,%f182
/*    265 */	fmuld,s	%f164,%f236,%f206


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f190,%f190,%f168,%f190
/*    265 */	fmaddd,s	%f112,%f116,%f112,%f112


/*    265 */	sxar2
/*    265 */	fand,s	%f110,%f48,%f110
/*    265 */	fmuld,s	%f132,%f132,%f198


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f174,%f192,%f174,%f174
/*    265 */	fnmsubd,s	%f138,%f196,%f236,%f196


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f176,%f194,%f176,%f176
/*    265 */	fmuld,s	%f200,%f200,%f212


/*    265 */	sxar2
/*    265 */	fmuld,s	%f202,%f202,%f214
/*    265 */	frsqrtad,s	%f182,%f216


/*    265 */	sxar2
/*    265 */	fmuld,s	%f182,%f236,%f220
/*    265 */	frsqrtad,s	%f190,%f218


/*    265 */	sxar2
/*    265 */	fmuld,s	%f190,%f236,%f222
/*    265 */	fand,s	%f112,%f54,%f112


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f308,%f110,%f184,%f110
/*    265 */	fnmsubd,s	%f140,%f198,%f236,%f198


/*    265 */	sxar2
/*    265 */	fmuld,s	%f174,%f174,%f224
/*    265 */	fmuld,s	%f176,%f176,%f226


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f130,%f196,%f130,%f130
/*    265 */	fnmsubd,s	%f204,%f212,%f236,%f212


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f206,%f214,%f236,%f214
/*    265 */	fmuld,s	%f120,%f120,%f210


/*    265 */	sxar2
/*    265 */	fmuld,s	%f216,%f216,%f228
/*    265 */	fnmsubd,s	%f122,%f208,%f236,%f122


/*    265 */	sxar2
/*    265 */	fmuld,s	%f218,%f218,%f230
/*    265 */	fcmplted,s	%f72,%f66,%f66


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f308,%f112,%f186,%f52
/*    265 */	fmaddd,s	%f132,%f198,%f132,%f132


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f178,%f224,%f236,%f224
/*    265 */	fnmsubd,s	%f180,%f226,%f236,%f226


/*    265 */	sxar2
/*    265 */	fmuld,s	%f130,%f130,%f232
/*    265 */	fmaddd,s	%f200,%f212,%f200,%f200


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f202,%f214,%f202,%f202
/*    265 */	fnmsubd,s	%f124,%f210,%f236,%f124


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f220,%f228,%f236,%f228
/*    265 */	fmaddd,s	%f118,%f122,%f118,%f118


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f222,%f230,%f236,%f230
/*    265 */	fcmplted,s	%f74,%f70,%f70


/*    265 */	sxar2
/*    265 */	fcmplted,s	%f72,%f84,%f84
/*    265 */	fmuld,s	%f132,%f132,%f234


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f174,%f224,%f174,%f174
/*    265 */	fmaddd,s	%f176,%f226,%f176,%f176


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f138,%f232,%f236,%f138
/*    265 */	fmuld,s	%f200,%f200,%f238


/*    265 */	sxar2
/*    265 */	fmuld,s	%f202,%f202,%f240
/*    265 */	fmaddd,s	%f120,%f124,%f120,%f120


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f216,%f228,%f216,%f184
/*    265 */	fand,s	%f118,%f66,%f118


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f218,%f230,%f218,%f186
/*    265 */	fcmplted,s	%f74,%f88,%f88


/*    265 */	sxar2
/*    265 */	fcmplted,s	%f72,%f104,%f104
/*    265 */	fnmsubd,s	%f140,%f234,%f236,%f140


/*    265 */	sxar2
/*    265 */	fmuld,s	%f174,%f174,%f242
/*    265 */	fmuld,s	%f176,%f176,%f244


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f130,%f138,%f130,%f130
/*    265 */	fnmsubd,s	%f204,%f238,%f236,%f238


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f206,%f240,%f236,%f240
/*    265 */	fand,s	%f120,%f70,%f120


/*    265 */	sxar2
/*    265 */	fmuld,s	%f184,%f184,%f246
/*    265 */	fnmsubd,sc	%f324,%f118,%f110,%f118


/*    265 */	sxar2
/*    265 */	fmuld,s	%f186,%f186,%f248
/*    265 */	fcmplted,s	%f74,%f108,%f108


/*    265 */	sxar2
/*    265 */	fcmplted,s	%f72,%f160,%f160
/*    265 */	fmaddd,s	%f132,%f140,%f132,%f132


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f178,%f242,%f236,%f178
/*    265 */	fnmsubd,s	%f180,%f244,%f236,%f180


/*    265 */	sxar2
/*    265 */	fand,s	%f130,%f84,%f130
/*    265 */	fmaddd,s	%f200,%f238,%f200,%f200


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f202,%f240,%f202,%f202
/*    265 */	fnmsubd,sc	%f324,%f120,%f52,%f68


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f220,%f246,%f236,%f246
/*    265 */	fnmsubd,s	%f222,%f248,%f236,%f248


/*    265 */	sxar2
/*    265 */	fcmplted,s	%f74,%f164,%f164
/*    265 */	fcmplted,s	%f72,%f182,%f182


/*    265 */	sxar2
/*    265 */	fcmplted,s	%f74,%f190,%f190
/*    265 */	fand,s	%f132,%f88,%f132


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f174,%f178,%f174,%f174
/*    265 */	fmaddd,s	%f176,%f180,%f176,%f176


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f342,%f130,%f118,%f130
/*    265 */	fmuld,s	%f200,%f200,%f250


/*    265 */	sxar2
/*    265 */	fmuld,s	%f202,%f202,%f252
/*    265 */	fmaddd,s	%f184,%f246,%f184,%f184


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f186,%f248,%f186,%f186
/*    265 */	fnmsubd,sc	%f342,%f132,%f68,%f86


/*    265 */	sxar2
/*    265 */	fand,s	%f174,%f104,%f174
/*    265 */	fand,s	%f176,%f108,%f176


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f204,%f250,%f236,%f204
/*    265 */	fnmsubd,s	%f206,%f252,%f236,%f206


/*    265 */	sxar2
/*    265 */	fmuld,s	%f184,%f184,%f254
/*    265 */	fmuld,s	%f186,%f186,%f34


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f362,%f174,%f130,%f174
/*    265 */	fnmsubd,sc	%f362,%f176,%f86,%f106


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f200,%f204,%f200,%f200
/*    265 */	fmaddd,s	%f202,%f206,%f202,%f202


/*    265 */	sxar2
/*    265 */	fnmsubd,s	%f220,%f254,%f236,%f220
/*    265 */	fnmsubd,s	%f222,%f34,%f236,%f222


/*    265 */	sxar2
/*    265 */	fand,s	%f200,%f160,%f200
/*    265 */	fand,s	%f202,%f164,%f202


/*    265 */	sxar2
/*    265 */	fmaddd,s	%f184,%f220,%f184,%f184
/*    265 */	fmaddd,s	%f186,%f222,%f186,%f186


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f418,%f200,%f174,%f200
/*    265 */	fnmsubd,sc	%f418,%f202,%f106,%f162


/*    265 */	sxar2
/*    265 */	fand,s	%f184,%f182,%f184
/*    265 */	fand,s	%f186,%f190,%f186


/*    265 */	sxar2
/*    265 */	fnmsubd,sc	%f444,%f184,%f200,%f184
/*    265 */	fnmsubd,sc	%f444,%f186,%f162,%f186

.L7112:


.L7111:


.L7114:


/*    149 */	sxar2
/*    149 */	add	%xg12,%xg10,%xg25
/* #00004 */	ldd,s	[%fp+255],%f244


/*     22 */	sxar2
/*     22 */	add	%xg4,%xg10,%xg26
/*     22 */	ldd,s	[%xg25],%f132


/*    277 */	sxar2
/*    277 */	ldd,s	[%xg25+16],%f140
/*    277 */	add	%xg10,64,%xg10


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


/*    276 */	sxar2
/*    276 */	fnmsubd,sc	%f430,%f184,%f144,%f184
/*    276 */	fnmsubd,sc	%f430,%f186,%f140,%f186

/*    277 */	bpos,pt	%icc, .L7114
	nop


.L7110:


.L7008:

/*    277 */	sxar1
/*    277 */	addcc	%xg8,1,%xg8

/*    277 */	bneg	.L7001
	nop


.L7009:

/*    277 */	sxar1
/*    277 */	ldx	[%i0+2199],%xg28

.L7014:


/*    149 */	sxar2
/*    149 */	add	%xg28,%xg10,%xg27
/* #00003 */	ldd,s	[%fp+255],%f240


/*     22 */	sxar2
/*     22 */	add	%xg10,32,%xg10
/*     22 */	ldd,s	[%xg27],%f200


/*    277 */	sxar2
/*    277 */	ldd,s	[%xg27+16],%f208
/*    277 */	subcc	%xg8,1,%xg8


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


/*    275 */	sxar2
/*    275 */	fand,s	%f220,%f210,%f220
/*    275 */	fnmsubd,sc	%f464,%f212,%f184,%f184

/*    276 */	sxar1
/*    276 */	fnmsubd,sc	%f464,%f220,%f186,%f186

/*    277 */	bpos,pt	%icc, .L7014
	nop


.L7010:


.L7001:


/*    277 */	sxar2
/*    277 */	std,s	%f184,[%fp+1263]
/*    277 */	std,s	%f186,[%fp+1279]

.L7002:


/*     22 */	sxar2
/*     22 */	add	%xg19,%xg1,%xg18
/*     22 */	ldd,s	[%fp+1263],%f236



/*    279 */	sxar2
/*    279 */	add	%xg1,32,%xg1
/*    279 */	add	%xg3,128,%xg3


/*     24 */	sxar2
/*     24 */	subcc	%xg0,1,%xg0
/*     24 */	std,s	%f236,[%xg18]


/*     25 */	sxar2
/*     25 */	ldd,s	[%fp+1279],%f238
/*     25 */	std,s	%f238,[%xg18+16]

/*    279 */	bne,pt	%icc, .L6996
/*    279 */	add	%o0,4,%o0


.L7003:

/*    279 */
/*    279 */	ba	.L6994
	nop


.L7005:

/*    279 *//*    279 */	call	__mpc_obar
/*    279 */	ldx	[%fp+2199],%o0

/*    279 *//*    279 */	call	__mpc_obar
/*    279 */	ldx	[%fp+2199],%o0


.L7006:

/*    279 */	ret
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
	.skip	1572864
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
	.byte	49
	.byte	48
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
.LB0..127.1:
	.skip	8
	.type	.LB0..127.1,#object
	.size	.LB0..127.1,.-.LB0..127.1
	.section	".rodata"
	.align	8
.LR0.cnt.10:
	.word	0X40100000,0
	.type	.LR0.cnt.10,#object
	.size	.LR0.cnt.10,.-.LR0.cnt.10
	.section	".rodata"
	.align	8
.LR0.cnt.9:
	.word	0X40080000,0
	.type	.LR0.cnt.9,#object
	.size	.LR0.cnt.9,.-.LR0.cnt.9
	.section	".rodata"
	.align	8
.LR0.cnt.8:
	.word	0X3FD00000,0
	.type	.LR0.cnt.8,#object
	.size	.LR0.cnt.8,.-.LR0.cnt.8
	.section	".rodata"
	.align	8
.LR0.cnt.7:
	.word	0X3FE00000,0
	.type	.LR0.cnt.7,#object
	.size	.LR0.cnt.7,.-.LR0.cnt.7
	.section	".rodata"
	.align	8
.LR0.cnt.6:
	.word	0X3FF00000,0
	.type	.LR0.cnt.6,#object
	.size	.LR0.cnt.6,.-.LR0.cnt.6
	.section	".rodata"
	.align	8
.LR0.cnt.5:
	.word	0,0
	.type	.LR0.cnt.5,#object
	.size	.LR0.cnt.5,.-.LR0.cnt.5
	.section	".rodata"
	.align	8
.LR0.cnt.4:
	.word	0X3FC99999,0X9999999A
	.type	.LR0.cnt.4,#object
	.size	.LR0.cnt.4,.-.LR0.cnt.4
	.section	".rodata"
	.align	8
.LR0.cnt.3:
	.word	0X3FC55555,0X55555555
	.type	.LR0.cnt.3,#object
	.size	.LR0.cnt.3,.-.LR0.cnt.3
	.section	".rodata"
	.align	8
.LR0.cnt.2:
	.word	0X3FD55555,0X55555555
	.type	.LR0.cnt.2,#object
	.size	.LR0.cnt.2,.-.LR0.cnt.2
	.section	".rodata"
	.align	8
.LR0.cnt.1:
	.word	0X3FC24924,0X92492492
	.type	.LR0.cnt.1,#object
	.size	.LR0.cnt.1,.-.LR0.cnt.1
	.section	".data"
	.align	16
.LS0:
	.align	8
.LS0.cnt.11:
	.word	1065353216
	.type	.LS0.cnt.11,#object
	.size	.LS0.cnt.11,.-.LS0.cnt.11
