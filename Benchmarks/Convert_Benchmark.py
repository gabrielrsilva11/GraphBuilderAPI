# import openpyxl
# #Compare subjects, predicates, objects
# wb_obj = openpyxl.load_workbook("Results/CaRB_Processed_LLM.xlsx")
# sheet_obj = wb_obj.active
# tsv_file = open("CaRB_bench.txt", "w")
# previous_id = 0
# for i in range(2, sheet_obj.max_row):
#     current_id = sheet_obj.cell(row=i, column=1).value
#     if previous_id != current_id:
#         current_sentence = sheet_obj.cell(row=i, column=2).value
#         previous_id = current_id
#     else:
#         subject = sheet_obj.cell(row=i, column=3).value
#         predicate = sheet_obj.cell(row=i, column=4).value
#         object = sheet_obj.cell(row=i, column=5).value
#         tsv_file.write("{0}\t{1}\t{2}\t{3}\n".format(current_sentence, subject, predicate, object))
#     if i == 20:
#         break
# tsv_file.close()