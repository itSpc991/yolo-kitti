"use client";

import { useState } from "react";
import styles from "./page.module.css";
import { Inter } from 'next/font/google';

const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
});

export default function Home() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
      setResult(null);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("请选择一个文件");
      return;
    }

    setIsProcessing(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/process", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("处理失败，请重试");
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error:", error);
      setError(error.message || "处理文件时出错");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className={`${styles.page} ${inter.className}`}>
      <main className={styles.main}>
        <h1 className={styles.title}>智能车辆视觉感知系统</h1>
        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.uploadContainer}>
            <input
              type="file"
              accept="image/*,video/*"
              onChange={handleFileChange}
              className={styles.fileInput}
              disabled={isProcessing}
            />
            <button
              type="submit"
              disabled={!file || isProcessing}
              className={styles.submitButton}
            >
              {isProcessing ? (
                <span className={styles.loading}>
                  <span className={styles.loadingDot}></span>
                  <span className={styles.loadingDot}></span>
                  <span className={styles.loadingDot}></span>
                </span>
              ) : (
                "开始处理"
              )}
            </button>
            {error && <p className={styles.error}>{error}</p>}
          </div>
        </form>

        {preview && (
          <div className={styles.previewContainer}>
            <h2 className={styles.resultTitle}>预览</h2>
            {file.type.startsWith("image/") ? (
              <img src={preview} alt="预览" className={styles.preview} />
            ) : (
              <video
                src={preview}
                controls
                className={styles.preview}
              />
            )}
          </div>
        )}

        {result && (
          <div className={styles.resultContainer}>
            <h2 className={styles.resultTitle}>处理结果</h2>
            {result.type === "image" ? (
              <img src={result.url} alt="处理结果" className={styles.result} />
            ) : (
              <video
                src={result.url}
                controls
                className={styles.result}
              />
            )}
          </div>
        )}
      </main>
    </div>
  );
}
