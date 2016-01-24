import subprocess
import os

def cal_sentence_blue(trans_lst, ref_lst, **kwargs):
    work_dir = kwargs['work_dir']
    trans = open('%s/trans'%work_dir, 'w')
    ref = open('%s/ref'%work_dir, 'w')
    trans.writelines(trans_lst)
    ref.writelines(ref_lst)
    child = subprocess.Popen(['sh', 'run.sh'], cwd=work_dir, shell=True)
    child.wait()

    risk_file = open('%s/BLEU-seg.scr'%work_dir, 'r')
    risks = [float(item.strip().split()[-1]) for item in risk_file]
    risk_file.close()

    return risks


if __name__ =='__main__':

    import configurations
    config = getattr(configurations, 'get_config_cs2en')()

    tr =['an', 'bn', 'c', 'd', 'r']
    ref =['a', 'bn', 'c', 'z', 'r']

