# -*- coding: utf-8 -*-

# 首先，您需要安裝一個必要的函式庫：pdf2docx
# 請在您的終端機或命令提示字元中執行以下指令：
# pip install pdf2docx

from pdf2docx import Converter
import os

def convert_pdf_to_docx(pdf_path, docx_path):
    """
    將指定的 PDF 檔案轉換為 DOCX 檔案。

    :param pdf_path: 輸入的 PDF 檔案路徑。
    :param docx_path: 輸出的 DOCX 檔案路徑。
    """
    try:
        # 檢查 PDF 檔案是否存在
        if not os.path.exists(pdf_path):
            print(f"錯誤：找不到 PDF 檔案 -> {pdf_path}")
            return

        print(f"讀取 PDF 檔案：'{os.path.basename(pdf_path)}'...")
        
        # 建立 Converter 物件
        cv = Converter(pdf_path)
        
        # 執行轉換，可以指定頁面範圍，例如 pages=[0, 1] 表示轉換前兩頁
        # 若要轉換所有頁面，則不需要傳入 pages 參數
        print("開始轉換，請稍候...")
        cv.convert(docx_path, start=0, end=None)
        
        # 關閉 Converter 物件
        cv.close()
        
        print(f"成功！檔案已儲存為：'{os.path.basename(docx_path)}'")

    except Exception as e:
        print(f"轉換過程中發生錯誤：{e}")

# --- 主程式執行區 ---
if __name__ == '__main__':
    # --- 請在此修改您的檔案路徑 ---

    # 1. 輸入的 PDF 檔案名稱
    # 請確保此 PDF 檔案與這個 Python 程式放在同一個資料夾下，
    # 或者提供完整的檔案路徑。
    pdf_file = r'D:\github\iSpan_python-FE_DM-cookbooks\course_slides\特徵工程系統化分析指南.pdf' 

    # 2. 輸出的 Word 檔案名稱
    # 轉換後的 .docx 檔案將會以此名稱儲存。
    docx_file = r'D:\github\iSpan_python-FE_DM-cookbooks\course_slides\特徵工程系統化分析指南.docx'
    
    # --- 修改結束 ---
    
    # 執行轉換函式
    convert_pdf_to_docx(pdf_file, docx_file)


    