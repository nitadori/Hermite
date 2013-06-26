exe=mic-8th
for snap in pl1k pl2k pl4k pl8k pl16k pl32k pl64k
do
    cp ${snap}.dat inp.dat
    ./${exe} < inp.8th | grep '##' >> ${exe}.log
done
