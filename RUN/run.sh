for order in 4th 6th 8th
do
  exe=hpc-${order}
  for snap in pl1k pl2k pl4k pl8k pl16k pl32k pl64k
  do  
      rm -f inp.dat
      ln -s  ${snap}.dat inp.dat
      ./${exe} < inp.${order} | grep '##' >> ${exe}.log
  done
done
