
import { useState } from "react";
import { Search, Plus, User, FileText } from "lucide-react";
import Sidebar from "@/components/Sidebar";
import DocumentList from "@/components/DocumentList";
import { mockDocuments } from "@/data/mockData";

type TabType = 'recent' | 'favorite';

export default function Home() {
  const [activeSection, setActiveSection] = useState("home");
  const [activeTab, setActiveTab] = useState&lt;TabType&gt;('recent');

  return (
    <div className="flex h-screen bg-white">
      <Sidebar
        activeSection={activeSection} onSectionChange={setActiveSection} />
      <main className="flex-1 flex flex-col overflow-hidden">
        <header className="border-b border-gray-200 px-8 py-6 flex items-center justify-between">
          <h1 className="text-4xl font-bold text-gray-800">主页</h1>
          <div className="flex items-center gap-6">
            <button className="p-3 hover:bg-gray-100 rounded-lg transition-colors">
              <Search className="w-8 h-8 text-gray-500" />
            </button>
            <button className="p-3 hover:bg-gray-100 rounded-lg transition-colors">
              <Plus className="w-8 h-8 text-gray-500" />
            </button>
            <div className="w-10 h-10 bg-yellow-400 rounded-full flex items-center justify-center">
              <User className="w-6 h-6 text-yellow-700" />
            </div>
          </div>
        </header>
        <div className="flex-1 overflow-y-auto px-8 py-8">
          <div className="mb-10">
            <button className="w-full max-w-sm border-2 border-gray-300 rounded-lg p-6 hover:border-blue-400 hover:bg-blue-50 transition-all flex items-center gap-4">
              <div className="w-14 h-14 bg-blue-600 rounded flex items-center justify-center">
                <div className="relative">
                  <FileText className="w-8 h-8 text-white" />
                  <Plus className="w-5 h-5 text-white absolute -bottom-1 -right-1" />
                </div>
              </div>
              <div className="text-left">
                <h3 className="text-xl font-bold text-gray-800">新建</h3>
                <p className="text-gray-600">新文档开始编辑</p>
              </div>
            </button>
          </div>

          <div className="flex gap-8 mb-8">
            <button
              onClick={() => setActiveTab('recent')}
              className={`text-xl font-semibold pb-2 border-b-4 transition-colors ${
                activeTab === 'recent' ? 'text-blue-600 border-blue-600' : 'text-gray-800 border-transparent'
              }`}
            >
              最近访问
            </button>
            <button
              onClick={() => setActiveTab('favorite')}
              className={`text-xl font-semibold pb-2 border-b-4 transition-colors ${
                activeTab === 'favorite' ? 'text-blue-600 border-blue-600' : 'text-gray-800 border-transparent'
              }`}
            >
              收藏
            </button>
          </div>

          <DocumentList documents={mockDocuments} />
        </div>
      </main>
    </div>
  );
}

