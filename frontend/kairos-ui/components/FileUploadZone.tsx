'use client';

import { useState, useCallback, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, Image, Code, Archive, Video, Music, File, X, CheckCircle, AlertCircle, Loader, Eye } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface FileUploadZoneProps {
  onFileDrop: (files: File[]) => void;
  className?: string;
  acceptedTypes?: string[];
  maxSize?: number;
  multiple?: boolean;
}

interface UploadedFile {
  file: File;
  id: string;
  status: 'uploading' | 'processing' | 'completed' | 'error';
  progress: number;
  analysis?: {
    type: string;
    summary: string;
    insights: string[];
    codeMetrics?: {
      language: string;
      lines: number;
      complexity: number;
    };
  };
  error?: string;
}

export function FileUploadZone({ 
  onFileDrop, 
  className = '',
  acceptedTypes = [],
  maxSize = 10 * 1024 * 1024, // 10MB default
  multiple = true 
}: FileUploadZoneProps) {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [dragActive, setDragActive] = useState(false);
  
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles = acceptedFiles.map(file => ({
      file,
      id: `${Date.now()}-${Math.random()}`,
      status: 'uploading' as const,
      progress: 0,
    }));

    setUploadedFiles(prev => [...prev, ...newFiles]);
    onFileDrop(acceptedFiles);

    // Simulate upload and processing
    newFiles.forEach(fileData => {
      simulateUpload(fileData.id);
    });
  }, [onFileDrop]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: acceptedTypes.length > 0 ? acceptedTypes.reduce((acc, type) => {
      acc[type] = [];
      return acc;
    }, {} as Record<string, string[]>) : undefined,
    maxSize,
    multiple,
    onDragEnter: () => setDragActive(true),
    onDragLeave: () => setDragActive(false),
  });

  const simulateUpload = (fileId: string) => {
    const file = uploadedFiles.find(f => f.id === fileId) || 
                 uploadedFiles[uploadedFiles.length - 1]; // Get the latest file as fallback
    
    let progress = 0;
    
    // Upload simulation
    const uploadInterval = setInterval(() => {
      progress += Math.random() * 30;
      
      setUploadedFiles(prev => prev.map(f => 
        f.id === fileId 
          ? { ...f, progress: Math.min(progress, 100) }
          : f
      ));
      
      if (progress >= 100) {
        clearInterval(uploadInterval);
        
        // Start processing
        setUploadedFiles(prev => prev.map(f => 
          f.id === fileId 
            ? { ...f, status: 'processing', progress: 0 }
            : f
        ));
        
        setTimeout(() => {
          // Complete processing with mock analysis
          const analysis = generateMockAnalysis(file?.file);
          
          setUploadedFiles(prev => prev.map(f => 
            f.id === fileId 
              ? { 
                  ...f, 
                  status: 'completed',
                  progress: 100,
                  analysis 
                }
              : f
          ));
        }, 2000 + Math.random() * 3000);
      }
    }, 200);
  };

  const generateMockAnalysis = (file?: File) => {
    if (!file) return undefined;
    
    const isCode = file.name.match(/\.(js|jsx|ts|tsx|py|java|cpp|c|html|css|json)$/);
    const isImage = file.type.startsWith('image/');
    const isText = file.type.startsWith('text/');
    
    if (isCode) {
      return {
        type: 'Code Analysis',
        summary: `Analyzed ${file.name} - ${getFileLanguage(file.name)} source code`,
        insights: [
          `File contains approximately ${Math.floor(Math.random() * 500) + 50} lines of code`,
          `Code complexity: ${['Low', 'Medium', 'High'][Math.floor(Math.random() * 3)]}`,
          'Follows modern coding standards',
          'No critical security issues detected'
        ],
        codeMetrics: {
          language: getFileLanguage(file.name),
          lines: Math.floor(Math.random() * 500) + 50,
          complexity: Math.random() * 10
        }
      };
    } else if (isImage) {
      return {
        type: 'Image Analysis',
        summary: `Processed ${file.name} - Image analysis complete`,
        insights: [
          `Resolution: ${Math.floor(Math.random() * 1920) + 800}x${Math.floor(Math.random() * 1080) + 600}`,
          `File size: ${formatFileSize(file.size)}`,
          'Image quality: High',
          'Contains no sensitive information'
        ]
      };
    } else {
      return {
        type: 'Document Analysis',
        summary: `Analyzed ${file.name} - Document processed successfully`,
        insights: [
          `File type: ${file.type || 'Unknown'}`,
          `Size: ${formatFileSize(file.size)}`,
          'Content appears to be safe',
          'Ready for agent analysis'
        ]
      };
    }
  };

  const getFileLanguage = (filename: string): string => {
    const ext = filename.split('.').pop()?.toLowerCase();
    const langMap: Record<string, string> = {
      js: 'JavaScript',
      jsx: 'React JSX',
      ts: 'TypeScript',
      tsx: 'React TSX',
      py: 'Python',
      java: 'Java',
      cpp: 'C++',
      c: 'C',
      html: 'HTML',
      css: 'CSS',
      json: 'JSON'
    };
    return langMap[ext || ''] || 'Unknown';
  };

  const getFileIcon = (file: File) => {
    if (file.type.startsWith('image/')) return <Image size={24} className="text-purple-500" />;
    if (file.type.startsWith('video/')) return <Video size={24} className="text-red-500" />;
    if (file.type.startsWith('audio/')) return <Music size={24} className="text-green-500" />;
    if (file.name.match(/\.(js|jsx|ts|tsx|py|java|cpp|c|html|css|json)$/)) return <Code size={24} className="text-blue-500" />;
    if (file.type.includes('zip') || file.type.includes('archive')) return <Archive size={24} className="text-orange-500" />;
    if (file.type.includes('text') || file.type.includes('document')) return <FileText size={24} className="text-gray-500" />;
    return <File size={24} className="text-gray-400" />;
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const removeFile = (fileId: string) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== fileId));
  };

  const getStatusIcon = (status: UploadedFile['status']) => {
    switch (status) {
      case 'uploading':
      case 'processing':
        return <Loader className="animate-spin text-blue-500" size={16} />;
      case 'completed':
        return <CheckCircle className="text-green-500" size={16} />;
      case 'error':
        return <AlertCircle className="text-red-500" size={16} />;
    }
  };

  const getStatusText = (file: UploadedFile) => {
    switch (file.status) {
      case 'uploading':
        return `Uploading... ${file.progress.toFixed(0)}%`;
      case 'processing':
        return 'Processing with AI agents...';
      case 'completed':
        return 'Analysis complete';
      case 'error':
        return file.error || 'Upload failed';
    }
  };

  return (
    <div className={`p-6 space-y-6 ${className}`}>
      {/* Header */}
      <div>
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">File Analysis</h2>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          Upload files for AI-powered analysis. Supports code, documents, images, and more.
        </p>
      </div>

      {/* Drop Zone */}
      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
          ${isDragActive || dragActive 
            ? 'border-kairos-400 bg-kairos-50 dark:bg-kairos-900/20' 
            : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
          }
        `}
      >
        <input {...getInputProps()} />
        <motion.div
          animate={{ scale: isDragActive ? 1.05 : 1 }}
          transition={{ type: "spring", stiffness: 300, damping: 20 }}
        >
          <Upload size={48} className={`mx-auto mb-4 ${
            isDragActive ? 'text-kairos-500' : 'text-gray-400'
          }`} />
          {isDragActive ? (
            <div>
              <p className="text-lg font-medium text-kairos-600 dark:text-kairos-400">Drop files here!</p>
              <p className="text-sm text-kairos-500 dark:text-kairos-300 mt-1">
                Files will be analyzed by AI agents
              </p>
            </div>
          ) : (
            <div>
              <p className="text-lg font-medium text-gray-900 dark:text-white">
                Drag & drop files here, or click to select
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
                Supports all file types • Max {formatFileSize(maxSize)} per file
              </p>
              <div className="flex items-center justify-center space-x-4 mt-4 text-xs text-gray-400">
                <div className="flex items-center space-x-1">
                  <Code size={16} />
                  <span>Code</span>
                </div>
                <div className="flex items-center space-x-1">
                  <FileText size={16} />
                  <span>Docs</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Image size={16} />
                  <span>Images</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Archive size={16} />
                  <span>Archives</span>
                </div>
              </div>
            </div>
          )}
        </motion.div>
      </div>

      {/* Uploaded Files */}
      {uploadedFiles.length > 0 && (
        <div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Uploaded Files ({uploadedFiles.length})
          </h3>
          <div className="space-y-3">
            <AnimatePresence>
              {uploadedFiles.map((fileData) => (
                <motion.div
                  key={fileData.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-3 flex-1">
                      {getFileIcon(fileData.file)}
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                          {fileData.file.name}
                        </p>
                        <div className="flex items-center space-x-3 mt-1">
                          <span className="text-xs text-gray-500">
                            {formatFileSize(fileData.file.size)}
                          </span>
                          <div className="flex items-center space-x-1">
                            {getStatusIcon(fileData.status)}
                            <span className="text-xs text-gray-600 dark:text-gray-300">
                              {getStatusText(fileData)}
                            </span>
                          </div>
                        </div>
                        
                        {/* Progress Bar */}
                        {(fileData.status === 'uploading' || fileData.status === 'processing') && (
                          <div className="mt-2">
                            <div className="bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                              <motion.div
                                className="bg-kairos-500 h-2 rounded-full"
                                initial={{ width: 0 }}
                                animate={{ width: `${fileData.progress}%` }}
                                transition={{ duration: 0.5 }}
                              />
                            </div>
                          </div>
                        )}

                        {/* Analysis Results */}
                        {fileData.analysis && fileData.status === 'completed' && (
                          <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            className="mt-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
                          >
                            <div className="flex items-center justify-between mb-2">
                              <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                                {fileData.analysis.type}
                              </h4>
                              <button className="text-kairos-500 hover:text-kairos-600 text-xs flex items-center space-x-1">
                                <Eye size={12} />
                                <span>View Full Analysis</span>
                              </button>
                            </div>
                            <p className="text-xs text-gray-600 dark:text-gray-300 mb-2">
                              {fileData.analysis.summary}
                            </p>
                            <div className="space-y-1">
                              {fileData.analysis.insights.slice(0, 2).map((insight, index) => (
                                <p key={index} className="text-xs text-gray-500 dark:text-gray-400">
                                  • {insight}
                                </p>
                              ))}
                            </div>
                          </motion.div>
                        )}
                      </div>
                    </div>
                    
                    <button
                      onClick={() => removeFile(fileData.id)}
                      className="ml-3 text-gray-400 hover:text-red-500 transition-colors"
                    >
                      <X size={16} />
                    </button>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>
      )}
    </div>
  );
}