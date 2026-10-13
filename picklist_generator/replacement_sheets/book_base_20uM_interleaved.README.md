# Interleaved 20 µM base plate

Built from `IDT order_06012026_Ref DNA Book_new source plate.xlsx`, using its four 96-well order worksheets and the interleaving diagram supplied on October 13, 2026.

| IDT source | 384-well rows | 384-well columns | Example |
| --- | --- | --- | --- |
| 1 (blue) | A,C,E,G,I,K,M,O | 01–12 | B01 → C01 |
| 2 (red) | A,C,E,G,I,K,M,O | 13–24 | B01 → C13 |
| 3 (yellow) | B,D,F,H,J,L,N,P | 01–12 | A01 → B01 |
| 4 (green) | B,D,F,H,J,L,N,P | 13–24 | A01 → B13 |

The CSV contains exactly the 252 ordered sequences, including 18 balanced rigid hinges and eight updated staples. Unordered wells are omitted. The plate is still `SourcePlate8[8]`, with 20 µM stocks. The existing non-interleaved CSV is preserved.

`Well Position` is the actual interleaved source location. `Original Well Position` is the logical replacement key used by PAINT sheets; it is not necessarily the previous physical location. Existing sequence matches retain their logical keys. Eight newly ordered sequences identify their logical keys in their names (H05, H08, H11, H14, I05, I08, I11, I14). IDT source plate, source well, and previous quadrant well columns retain provenance. All sequences are taken directly from the workbook.

Restart the app and select **LowConc base plate — 20 µM (interleaved)** under **Base plate stock** in RGB Image or Run & Mixing Settings. This selects the ordered rigid hinges. Flexible hinges are absent from this order; choosing them is rejected rather than silently omitting hinges.

Prepare and save a new image batch after selecting this plate. Previously saved picklists retain their old well locations. PAINT replacement plate 7 is unchanged.

To rebuild (requires openpyxl), from the picklist_generator folder:

```sh
python3 scripts/build_interleaved_base.py '/path/to/IDT order_06012026_Ref DNA Book_new source plate.xlsx' replacement_sheets/book_base_20uM.csv replacement_sheets/book_base_20uM_interleaved.csv
```
