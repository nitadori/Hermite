arch=avx2
for snap in pl64k pl32k pl16k pl8k pl4k pl2k pl1k
do  
  rm -f inp.dat
  ln -s  ${snap}.dat inp.dat
  for order in 4th 6th 8th
  do
    exe=${arch}-${order}
    ./${exe} < inp.${order} 2>>  ${exe}.err 1>> ${exe}.log
  done
done
