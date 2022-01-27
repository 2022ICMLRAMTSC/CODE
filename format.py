# noqa
import subprocess


def shell_exec(command_as_string):
    """
    Executes the command in shell script. Avoid using the yes | command pattern as that seems to cause an out of memory
    issue.

    Args:
        command_as_string: just a string, as if you were typing it into a  terminal

    Returns:
        stdout, stderr
    """
    try:

        process = subprocess.Popen(
            command_as_string.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False
        )
        stdout, stderr = process.communicate()

    except subprocess.CalledProcessError as e:
        stdout = e.stdout
        stderr = e.stderr

    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")

    return stdout, stderr


stdout, stderr = shell_exec("python3 -m black .")
print(stdout)
print(stderr)

print("Checking Flake 8")
stdout, stderr = shell_exec("python3 -m flake8 .")
print(stdout)
print(stderr)
