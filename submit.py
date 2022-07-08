import os
from time import sleep
import sys
import subprocess

from contextlib import redirect_stdout

with open('submitLog.txt', 'w') as logf:
    with redirect_stdout(logf):
        # submits jobs to cluster based on variations of a template sh script

        fname = 'tExp.sh'
        if len(sys.argv) >= 2:
            fname = sys.argv[1]

        with open(fname, 'r') as file:
            data = file.read()

        print(data)

        prefix = "test1_n#_t#_arch#_trans#_w#"

        # ns = [1,2,3,4,6,8,12,16]
        # partition sizes
        ns = [16, 12, 8, 6, 4, 3, 2, 1]

        # replace existing model files.
        # If false this will continue after interruption without redoing work.
        if len(sys.argv) >= 3:
            replaceEx = sys.argv[2] == "-r"
        else:
            replaceEx = False

        i = 0

        # if true runs max(min(n, numTrials), numTrialsRep) trials at dataset size 1/n.
        # If n < numTrialsRep some trials will use the same partition.
        # If false run min(n, numTrials) times
        # All trials will have different partitions
        repeatTrial = False
        numTrials = 1  # 16
        numTrialsRep = 1  # 4

        # use data augmentation or not
        # transform "ResNet"
        # transform "ResNetNoAugment"

        # weightDecay 3e-4

        saveFreq = 50

        seed = None

        for arch in ["PreResNet20", "PreResNet20_4x"]:
            for transform in ["ResNet"]:
                for weightDecay in [0]:
                    for trialNum in range(0, numTrials):
                        for n in ns:
                            if (trialNum >= n and not repeatTrial) or (trialNum >= n and trialNum >= numTrialsRep):
                                continue

                            dataTrial = trialNum % n
                            # print("submit")

                            tprefix = prefix.replace("t#", "t" + str(trialNum))
                            tprefix = tprefix.replace("trans#", transform)
                            tprefix = tprefix.replace("arch#", arch)
                            tprefix = tprefix.replace("n#", "n" + str(n))
                            tprefix = tprefix.replace("w#", "w" + str(weightDecay))


                            # # gets jobs currently in queue
                            result = subprocess.run(
                                ['squeue', '--format="%.18i %.9P %j %.2t %.10M %.6D %R"', '-u', 'meinhorn6'],
                                capture_output=True, text=True).stdout
                            strOut = result
                            # check if this experiment has already been ran, or is currently running
                            if not replaceEx and (os.path.isdir("history/" + tprefix) or (
                                    tprefix + "-n" + str(n) in strOut)):
                                continue

                            tdata = data.replace("p#", tprefix)
                            tdata = tdata.replace("i#", str(i))
                            tdata = tdata.replace("n#", str(n))
                            tdata = tdata.replace("arch#", arch)
                            tdata = tdata.replace("trans#", transform)
                            tdata = tdata.replace("o#", str(dataTrial))
                            tdata = tdata.replace("w#", str(weightDecay))
                            tdata = tdata.replace("f#", str(saveFreq))

                            if seed is None:
                                tdata = tdata.replace("--seed=s#", "")
                            else:
                                tdata = tdata.replace("s#", str(seed))

                            print(tdata)
                            textfile = open("autoExp.sh", "w")
                            a = textfile.write(tdata)
                            textfile.close()
                            # os.system("sbatch autoExp.sh")
                            result = subprocess.run(['sbatch', 'autoExp.sh'], capture_output=True, text=True).stdout
                            strOut = result
                            print(strOut)

                            result = subprocess.run(
                                ['squeue', '--format="%.18i %.9P %j %.2t %.10M %.6D %R"', '-u', 'meinhorn6'],
                                capture_output=True, text=True).stdout
                            strOut = result
                            print(strOut)
                            nlines = strOut.count('\n')
                            print(nlines)

                            # limits concurrent execution by checking the number of lines in the queue
                            while nlines > 15:
                                result = subprocess.run(
                                    ['squeue', '--format="%.18i %.9P %j %.2t %.10M %.6D %R"', '-u', 'meinhorn6'],
                                    capture_output=True, text=True).stdout
                                strOut = result

                                print(strOut)
                                nlines = strOut.count('\n')
                                print(nlines)
                                sleep(900)
                            sleep(1)
                            i += 1
