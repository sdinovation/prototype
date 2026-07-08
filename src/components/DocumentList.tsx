
import { Document } from "@/types/document";
import { MoreVertical } from "lucide-react";

interface DocumentListProps {
  documents: Document[];
}

export default function DocumentList({ documents }: DocumentListProps) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="border-b-2 border-gray-200">
            <th className="text-left py-4 px-3 text-gray-500 font-semibold"></th>
            <th className="text-left py-4 px-3 text-gray-500 font-semibold">标题</th>
            <th className="text-left py-4 px-3 text-gray-500 font-semibold">位置</th>
            <th className="text-left py-4 px-3 text-gray-500 font-semibold">所有者</th>
            <th className="text-left py-4 px-3 text-gray-500 font-semibold">创建时间</th>
            <th className="text-left py-4 px-3 text-gray-500 font-semibold">最近访问</th>
            <th className="text-left py-4 px-3 text-gray-500 font-semibold"></th>
          </tr>
        </thead>
        <tbody>
          {documents.map((doc) => (
            <tr key={doc.id} className="border-b border-gray-100 hover:bg-gray-50 transition-colors">
              <td className="py-4 px-3">
                <input type="checkbox" className="w-4 h-4" />
              </td>
              <td className="py-4 px-3">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-blue-600 rounded flex items-center justify-center">
                    <span className="text-white text-lg">📄</span>
                  </div>
                  <div>
                    <div className="font-medium text-gray-800">{doc.title}</div>
                  </div>
                  {doc.isPinned && <span className="ml-2 text-gray-500">📌</span>}
                </div>
              </td>
              <td className="py-4 px-3 text-gray-700">{doc.location}</td>
              <td className="py-4 px-3 text-gray-700">{doc.owner}</td>
              <td className="py-4 px-3 text-gray-700">{doc.createdAt}</td>
              <td className="py-4 px-3 text-gray-700">{doc.lastAccessed}</td>
              <td className="py-4 px-3">
                <button className="p-2 hover:bg-gray-200 rounded transition-colors">
                  <MoreVertical className="w-5 h-5 text-gray-500" />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="flex items-center justify-center py-12 text-gray-500">
        <div className="w-48 h-px bg-gray-300"></div>
        <span className="mx-4">已经到底了</span>
        <div className="w-48 h-px bg-gray-300"></div>
      </div>
    </div>
  );
}

