// NUMBSKULL.exe: the double-click launcher in the portable download.
// It opens the control panel (files\app.py) with the Python bundled in
// files\python, without a console window. build_portable.py compiles it with
// the C# compiler that comes with Windows, filling in MAX_FOLDER_LENGTH.
using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

class Launcher
{
    // Windows can't load files whose full path is over 260 characters, and
    // some library files sit deep inside the files folder.
    const int MAX_FOLDER_LENGTH = __MAX_FOLDER_LENGTH__;

    [STAThread]
    static void Main()
    {
        string here = AppDomain.CurrentDomain.BaseDirectory;
        string files = Path.Combine(here, "files");
        string pythonw = Path.Combine(files, "python", "pythonw.exe");

        if (!File.Exists(pythonw))
        {
            MessageBox.Show(
                "NUMBSKULL can't find its files.\n\n" +
                "Extract the whole NUMBSKULL folder from the zip first (right-click the zip, " +
                "Extract All), and keep NUMBSKULL.exe next to its \"files\" folder.",
                "NUMBSKULL", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }
        if (here.Length > MAX_FOLDER_LENGTH)
        {
            MessageBox.Show(
                "This folder is too deep inside other folders for Windows:\n\n" + here + "\n\n" +
                "Move the NUMBSKULL folder somewhere shorter, for example straight into " +
                "Documents or C:\\NUMBSKULL, then open NUMBSKULL.exe again.",
                "NUMBSKULL", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return;
        }

        var start = new ProcessStartInfo(pythonw, "app.py");
        start.WorkingDirectory = files;
        start.UseShellExecute = false;
        Process.Start(start);
    }
}
