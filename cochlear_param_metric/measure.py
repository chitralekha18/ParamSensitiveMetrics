import subprocess
import sys
import numpy as np
import scipy.io as sio

def measStats(inpath, output_dir="./tmp"):
    """measStats
    This function will call MATLAB and measure the statistics from McWalter's 
    code.

    Parameters
    ----------
    inpath : str
        the audio path which need to be measured, or the directory path
        where containing all audio files need to be measured. 
    output_dir: str
        the output dir path. the name will remain the same as the audio file, 
        only the extention will be different. 
    """
    subprocess.call(f"matlab -batch \"Meas('{inpath}', '{output_dir}')\"")


def loadmat(path):
    """loadmat [summary]

    Parameters
    ----------
    path : str
        The path of a .mat file

    Returns
    -------
    dict
        the data in the .mat file
    """
    res = sio.loadmat(path)
    res.pop('__header__')
    res.pop('__version__')
    res.pop('__globals__')
    res.pop('I')
    return res

def matDist(mat1, mat2):
    """matDist
    Measure the cosine distance

    Parameters
    ----------
    mat1, mat2: np.array
        two matrix/array

    Returns
    -------
    float
        the cosine distance
    """
    v1, v2 = mat1.flatten(), mat2.flatten()
    cos = 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))    
    return cos

def soundDist(s1, s2):
    """soundDist 
    The distance of two .mat files
    
    Parameters
    ----------
    s1, s2 : str
        The .mat filepath

    Returns
    -------
    float
        the overall distance
    """
    m1, m2 = loadmat(s1), loadmat(s2)
    d = 0
    for e in m1:
        d += matDist(m1[e], m2[e])
    return d

def l2Float(l):
    """Transform a str list to a float list"""
    return [float(x) for x in l]

def l2Int(l):
    """Transform a str list to a int list"""
    return [int(x) for x in l]




"""
The following part is for calling the subprocess to call the MATLAB
Comes from: https://stackoverflow.com/a/59339154/11483738
"""

import signal
import subprocess as sp

class VerboseCalledProcessError(sp.CalledProcessError):
    def __str__(self):
        if self.returncode and self.returncode < 0:
            try:
                msg = "Command '%s' died with %r." % (
                    self.cmd, signal.Signals(-self.returncode))
            except ValueError:
                msg = "Command '%s' died with unknown signal %d." % (
                    self.cmd, -self.returncode)
        else:
            msg = "Command '%s' returned non-zero exit status %d." % (
                self.cmd, self.returncode)

        return f'{msg}\n' \
               f'Stdout:\n' \
               f'{self.output}\n' \
               f'Stderr:\n' \
               f'{self.stderr}'


def bash(cmd, print_stdout=True, print_stderr=True):
    proc = sp.Popen(cmd, stderr=sp.PIPE, stdout=sp.PIPE, shell=True, universal_newlines=True)

    all_stdout = []
    all_stderr = []
    while proc.poll() is None:
        for stdout_line in proc.stdout:
            if stdout_line != '':
                if print_stdout:
                    print(stdout_line, end='')
                all_stdout.append(stdout_line)
        for stderr_line in proc.stderr:
            if stderr_line != '':
                if print_stderr:
                    print(stderr_line, end='', file=sys.stderr)
                all_stderr.append(stderr_line)

    stdout_text = ''.join(all_stdout)
    stderr_text = ''.join(all_stderr)
    if proc.wait() != 0:
        raise VerboseCalledProcessError(proc.returncode, cmd, stdout_text, stderr_text)