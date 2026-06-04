
import { Search, Home, Menu, MoreVertical, Pin, Share2, Star, FileText, Trash2 } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { useDocumentStore } from "@/store/useDocumentStore";
import { Document } from "@/types/document";

interface SidebarProps {
  activeSection: string;
  onSectionChange: (section: string) =&gt; void;
  onDeleteDocument?: (doc: Document) =&gt; void;
}

interface DropdownState {
  open: boolean;
  docId: string | null;
}

export default function Sidebar({ activeSection, onSectionChange, onDeleteDocument }: SidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [pinnedCollapsed, setPinnedCollapsed] = useState(false);
  const [myDocsCollapsed, setMyDocsCollapsed] = useState(false);
  const [dropdown, setDropdown] = useState&lt;DropdownState&gt;({ open: false, docId: null });
  const dropdownRef = useRef&lt;HTMLDivElement&gt;(null);
  
  const { pinnedDocuments, myDocuments, togglePin } = useDocumentStore();

  const menuItems = [
    { id: "home", label: "主页", icon: Home },
  ];

  const toggleDropdown = (docId: string) =&gt; {
    setDropdown(prev =&gt; ({
      open: prev.docId === docId ? !prev.open : true,
      docId
    }));
  };

  useEffect(() =&gt; {
    const handleClickOutside = (event: MouseEvent) =&gt; {
      if (dropdownRef.current &amp;&amp; !dropdownRef.current.contains(event.target as Node)) {
        setDropdown({ open: false, docId: null });
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () =&gt; document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const renderDocItem = (doc: Document, section: string) =&gt; (
    &lt;div key={doc.id} className="relative group"&gt;
      &lt;button className="w-full flex items-center gap-3 px-4 py-2 text-left hover:bg-gray-100 rounded-lg transition-colors"&gt;
        &lt;div className="w-6 h-6 text-blue-500 flex items-center justify-center"&gt;
          &lt;FileText className="w-5 h-5" /&gt;
        &lt;/div&gt;
        &lt;span className="text-gray-700 text-sm flex-1"&gt;{doc.title}&lt;/span&gt;
        &lt;button
          onClick={(e) =&gt; {
            e.stopPropagation();
            toggleDropdown(doc.id);
          }}
          className="p-1 hover:bg-gray-200 rounded transition-colors opacity-0 group-hover:opacity-100"
        &gt;
          &lt;MoreVertical className="w-4 h-4 text-gray-500" /&gt;
        &lt;/button&gt;
      &lt;/button&gt;
      {dropdown.open &amp;&amp; dropdown.docId === doc.id &amp;&amp; (
        &lt;div
          ref={dropdownRef}
          className="absolute right-2 top-full mt-1 bg-white rounded-lg shadow-lg border border-gray-200 min-w-[180px] z-50"
        &gt;
          &lt;button
            onClick={() =&gt; {
              togglePin(doc.id);
              setDropdown({ open: false, docId: null });
            }}
            className={`w-full flex items-center gap-3 px-4 py-2 text-left transition-colors rounded-t-lg ${doc.isPinned ? 'text-red-500 bg-gray-100' : 'text-gray-700 hover:bg-gray-50'}`}
          &gt;
            &lt;Pin className="w-4 h-4" /&gt;
            &lt;span className="text-sm"&gt;{doc.isPinned ? '点击取消文件置顶' : '点击将文件置顶'}&lt;/span&gt;
          &lt;/button&gt;
          &lt;div className="border-t border-gray-100"&gt;&lt;/div&gt;
          &lt;button className="w-full flex items-center gap-3 px-4 py-2 text-left hover:bg-gray-50 transition-colors text-gray-700"&gt;
            &lt;Share2 className="w-4 h-4" /&gt;
            &lt;span className="text-sm"&gt;分享&lt;/span&gt;
          &lt;/button&gt;
          &lt;button className="w-full flex items-center gap-3 px-4 py-2 text-left hover:bg-gray-50 transition-colors text-amber-600"&gt;
            &lt;Star className="w-4 h-4" /&gt;
            &lt;span className="text-sm"&gt;从收藏中移除&lt;/span&gt;
          &lt;/button&gt;
          &lt;button className="w-full flex items-center gap-3 px-4 py-2 text-left hover:bg-gray-50 transition-colors text-gray-700"&gt;
            &lt;FileText className="w-4 h-4" /&gt;
            &lt;span className="text-sm"&gt;创建副本&lt;/span&gt;
          &lt;/button&gt;
          &lt;div className="border-t border-gray-100"&gt;&lt;/div&gt;
          &lt;button
            onClick={() =&gt; {
              if (onDeleteDocument) onDeleteDocument(doc);
              setDropdown({ open: false, docId: null });
            }}
            className="w-full flex items-center gap-3 px-4 py-2 text-left hover:bg-red-50 transition-colors text-red-500 rounded-b-lg"
          &gt;
            &lt;Trash2 className="w-4 h-4" /&gt;
            &lt;span className="text-sm"&gt;删除&lt;/span&gt;
          &lt;/button&gt;
        &lt;/div&gt;
      )}
    &lt;/div&gt;
  );

  return (
    &lt;aside className={`h-screen bg-white border-r border-gray-200 flex flex-col transition-all duration-300 ${isCollapsed ? 'w-16' : 'w-60'}`}&gt;
      &lt;div className="p-4 border-b border-gray-200 flex items-center justify-between"&gt;
        {!isCollapsed &amp;&amp; (
          &lt;div className="flex items-center gap-2"&gt;
            &lt;div className="w-7 h-7 bg-gradient-to-br from-blue-500 to-purple-600 rounded-md flex items-center justify-center"&gt;
              &lt;span className="text-white text-base"&gt;☁&lt;/span&gt;
            &lt;/div&gt;
            &lt;span className="font-medium text-gray-800"&gt;雷云云文档&lt;/span&gt;
          &lt;/div&gt;
        )}
        &lt;button
          onClick={() =&gt; setIsCollapsed(!isCollapsed)}
          className="p-2 hover:bg-gray-100 rounded transition-colors"
        &gt;
          &lt;Menu className="w-5 h-5 text-gray-600" /&gt;
        &lt;/button&gt;
      &lt;/div&gt;

      {!isCollapsed &amp;&amp; (
        &lt;div className="p-4"&gt;
          &lt;div className="flex items-center gap-2 text-gray-500 bg-gray-100 p-2 rounded"&gt;
            &lt;Search className="w-4 h-4" /&gt;
            &lt;span className="text-sm"&gt;搜索&lt;/span&gt;
          &lt;/div&gt;
        &lt;/div&gt;
      )}

      &lt;nav className="flex-1 overflow-y-auto p-2 space-y-1"&gt;
        {menuItems.map((item) =&gt; {
          const Icon = item.icon;
          const isActive = activeSection === item.id;
          return (
            &lt;button
              key={item.id}
              onClick={() =&gt; onSectionChange(item.id)}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-md transition-all ${
                isActive
                  ? 'bg-blue-50 text-blue-600'
                  : 'text-gray-600 hover:bg-gray-50'
              }`}
            &gt;
              &lt;Icon className="w-5 h-5" /&gt;
              {!isCollapsed &amp;&amp; &lt;span className="text-sm font-medium"&gt;{item.label}&lt;/span&gt;}
            &lt;/button&gt;
          );
        })}

        {!isCollapsed &amp;&amp; (
          &lt;div className="mt-4 space-y-4"&gt;
            &lt;div&gt;
              &lt