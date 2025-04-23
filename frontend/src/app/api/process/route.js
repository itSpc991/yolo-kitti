import { NextResponse } from "next/server";
import { writeFile, unlink } from "fs/promises";
import { exec } from "child_process";
import { promisify } from "util";
import path from "path";
import { mkdir } from "fs/promises";
import os from "os";
import { readdir } from "fs/promises";

const execAsync = promisify(exec);

// 清理旧文件
async function cleanupOldFiles(directory, currentFile) {
  try {
    const files = await readdir(directory);
    for (const file of files) {
      if (file !== path.basename(currentFile)) {
        const filePath = path.join(directory, file);
        await unlink(filePath);
        console.log(`已删除旧文件: ${filePath}`);
      }
    }
  } catch (error) {
    console.error(`清理文件时出错: ${error}`);
  }
}

export const config = {
  api: {
    bodyParser: false,
  },
};

export async function POST(request) {
  try {
    console.log("开始处理文件上传请求...");

    // 1. 创建必要的目录
    const uploadDir = path.join(process.cwd(), "public", "uploads");
    const resultDir = path.join(process.cwd(), "public", "results");
    console.log("创建目录:", { uploadDir, resultDir });

    try {
      await mkdir(uploadDir, { recursive: true });
      await mkdir(resultDir, { recursive: true });
      // 确保目录有正确的权限
      await execAsync(`chmod 777 "${uploadDir}" "${resultDir}"`);
      console.log("目录创建成功并设置权限");
    } catch (error) {
      console.error("创建目录失败:", error);
      throw error;
    }

    // 2. 处理上传的文件
    const formData = await request.formData();
    const file = formData.get("file");

    if (!file) {
      console.error("没有接收到文件");
      return NextResponse.json(
        { error: "No file provided" },
        { status: 400 }
      );
    }

    // 检查文件类型
    if (!file.type.startsWith("video/") && !file.type.startsWith("image/")) {
      console.error("不支持的文件类型:", file.type);
      return NextResponse.json(
        { error: "Only video and image files are supported" },
        { status: 400 }
      );
    }

    console.log("文件信息:", {
      name: file.name,
      type: file.type,
      size: file.size
    });

    // 3. 保存上传的文件
    const timestamp = Date.now();
    const inputExt = file.type.startsWith("image/") ? ".jpg" : ".mp4";
    const outputExt = file.type.startsWith("image/") ? ".jpg" : ".mp4";
    const inputPath = path.join(uploadDir, `input_${timestamp}${inputExt}`);
    const outputPath = path.join(resultDir, `output_${timestamp}${outputExt}`);
    console.log("文件路径:", {
      inputPath,
      outputPath,
      cwd: process.cwd(),
      publicDir: path.join(process.cwd(), "public"),
      uploadsDir: path.join(process.cwd(), "public", "uploads"),
      resultsDir: path.join(process.cwd(), "public", "results")
    });

    try {
      const bytes = await file.arrayBuffer();
      const buffer = Buffer.from(bytes);
      await writeFile(inputPath, buffer);
      // 设置文件权限
      await execAsync(`chmod 666 "${inputPath}"`);
      console.log("文件保存成功:", inputPath);

      // 清理旧的输入文件
      await cleanupOldFiles(uploadDir, inputPath);
    } catch (error) {
      console.error("保存文件失败:", error);
      throw error;
    }

    // 4. 设置环境变量
    const projectRoot = path.resolve(process.cwd(), "..");
    const pythonPath = "/opt/homebrew/Caskroom/miniconda/base/envs/track/bin/python";
    const scriptPath = path.resolve(projectRoot, "track-frondend.py");
    console.log("路径信息:", { projectRoot, pythonPath, scriptPath });

    // 检查文件是否存在
    try {
      await execAsync(`test -f "${scriptPath}"`);
      console.log("Python 脚本存在");
    } catch (error) {
      console.error("Python 脚本不存在:", scriptPath);
      return NextResponse.json(
        { error: "Python script not found" },
        { status: 500 }
      );
    }

    // 设置完整的环境变量
    const env = {
      ...process.env,
      PATH: `/opt/homebrew/Caskroom/miniconda/base/envs/track/bin:${process.env.PATH}`,
      PYTHONPATH: `${projectRoot}:${process.env.PYTHONPATH || ''}`,
      CONDA_DEFAULT_ENV: 'track',
      CONDA_PREFIX: '/opt/homebrew/Caskroom/miniconda/base/envs/track',
      CONDA_SHLVL: '2',
      CONDA_PROMPT_MODIFIER: '(track) '
    };
    console.log("环境变量:", env);

    // 5. 处理文件
    console.log("处理文件...");
    let result;
    try {
      const command = `cd "${projectRoot}" && ${pythonPath} "${scriptPath}" "${inputPath}" "${outputPath}"`;
      console.log("执行命令:", command);
      const { stdout, stderr } = await execAsync(command, {
        env,
        maxBuffer: 1024 * 1024 * 10
      });

      console.log("Python 脚本输出:", stdout.substring(0, 1000));
      if (stderr) console.error("Python 脚本错误:", stderr.substring(0, 1000));

      // 检查输出文件是否存在
      try {
        await execAsync(`test -f "${outputPath}"`);
        console.log("输出文件已成功创建:", outputPath);
        // 设置输出文件权限
        await execAsync(`chmod 666 "${outputPath}"`);
        // 清理旧的输出文件
        await cleanupOldFiles(resultDir, outputPath);
      } catch (error) {
        console.error("输出文件未找到:", outputPath);
        // 检查目录内容
        const { stdout: lsOutput } = await execAsync(`ls -la "${resultDir}"`);
        console.log("结果目录内容:", lsOutput);
        throw new Error("输出文件未成功创建");
      }

      result = {
        type: file.type.startsWith("image/") ? "image" : "video",
        url: `/results/output_${timestamp}${outputExt}`,
      };
    } catch (error) {
      console.error("执行 Python 脚本时出错:", error);
      console.error("错误详情:", {
        message: error.message,
        code: error.code,
        signal: error.signal,
        stdout: error.stdout ? error.stdout.substring(0, 1000) : null,
        stderr: error.stderr ? error.stderr.substring(0, 1000) : null
      });
      throw error;
    }

    // 6. 返回处理结果
    console.log("处理完成，返回结果:", result);
    return NextResponse.json(result);
  } catch (error) {
    console.error("处理文件时出错:", error);
    console.error("错误堆栈:", error.stack);
    return NextResponse.json(
      { error: error.message || "Error processing file" },
      { status: 500 }
    );
  }
} 