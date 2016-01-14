./plain2sgm src trans > src.sgm
./plain2sgm ref ref > ref.sgm
./plain2sgm tst trans >trans.sgm
./mteval-v13.pl -b -d 2 --metricsMATR -s src.sgm -r ref.sgm -t trans.sgm >log.bleu
