import os

if __name__ == '__main__':
    exp_confs = os.listdir('./exp_confs')
    
    for exp_name in exp_confs:
        exp_name = exp_name.replace('.yaml','')
        if exp_name != 'default':
            command = "sbatch --export=exp_name="+exp_name+" template.job"
            os.system(command)
            print("deployed exp: ",exp_name)