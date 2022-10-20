import csv
import os
import csv
from datetime import datetime


if __name__ == '__main__':
    exp_confs = os.listdir('./exp_confs')
    
    with open('./exp_log.csv','a') as trng_log:
        csv_writer = csv.writer(trng_log)
        csv_writer.writerow(['datetime','job_id','exp_name'])
        for exp_name in exp_confs:
            exp_name = exp_name.replace('.yaml','')
            
            if exp_name != 'default' and ('tstng' not in exp_name):
                command = "sbatch --export=exp_name="+exp_name+" template.job"
                # os.system(command)
                job_id = os.popen(command).read().replace('Submitted batch job ','')         

                now = datetime.now()
                date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                # write the log
                csv_writer.writerow([date_time_str, job_id, exp_name])

                print("\tdeployed exp: ",exp_name)