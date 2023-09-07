import pandapower.networks as pn
import pandapower as pp
import pandas as pd
import os

from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, Border, Side

rename_sheet = {
    #reXplan        :#pp-net
    'generators'    : 'gen',
    'external_gen'  : 'ext_grid',
    'transformers'  : 'trafo',
    'nodes'         : 'bus',
    'switches'      : 'switch',
    'loads'         : 'load',
    'cost'          : 'poly_cost',
    'lines'         : 'line',
}

rename_column = {
    'from_bus'      : 'from_bus',
    'to_bus'        : 'to_bus',
}

def style_formatting(ws):
    for column in ws.columns:
        max_length = 0
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(cell.value)
            except:
                pass
        adjusted_width = (max_length + 3)
        ws.column_dimensions[column[0].column_letter].width = adjusted_width

    font = Font(bold=True)
    border = Border(bottom=Side(border_style="thick"))
    for cell in ws[1]:
        cell.font = font
        cell.border = border

    font = Font(bold=True)
    border = Border(bottom=Side(border_style="thick"))
    for cell in ws[1]:
        cell.font = font
        cell.border = border

def rename_element(sheet, column, values, net, rename = False):
    # TODO: handling of rename = True
    if values.empty:
        pass

    elif values.dtype == bool:
        values = values.astype('object').map({True: 'True', False: 'False'})

    elif column == 'name' and rename:
        if sheet == 'nodes':
            for number in values.index:
                values[number] = 'bus' + str(number + 1)
        else:
            values.reset_index(drop=True, inplace=True)
            for number in values.index:
                values[number] = rename_sheet[sheet] + str(number + 1)
            
    elif column == 'node' or column == 'node_p' or column == 'node_s' or column == 'from_bus' or column == 'to_bus': 

        if isinstance(net, pp.auxiliary.pandapowerNet):              
            bus_column = getattr(net, rename_sheet[sheet])[rename_column[column]]   # renamed column??
            bus_names = net.bus.loc[bus_column.tolist(), 'name']
            values = bus_names.reset_index(drop=True)
            values = values.rename(column)
        else:
            raise TypeError('Provided datatype of network is not compliant')

    else:
        pass    
        # print(f"No need to rename for: [{sheet}] - [{column}]")

    return values

def import_grid(net, rename = False):
    """
	Creates a reXplan compliant network as excel file from pandapower.

    INPUT:
		net (dict) - pandapower format network
        rename (bool) - False: Naming as saved in pandapower net; True: Elements renamed

    EXAMPLE:
		>>> import_grid(net)
		>>> import_grid(pn.case14())
    """

    path = os.path.dirname(os.getcwd())
    fields_maps = pd.read_csv(os.path.join(path + "\\reXplan",'fields_map.csv'))
    dfs_dict = pd.read_excel('template.xlsx', sheet_name=None)

    for index, row in fields_maps.iterrows():
        value = row.iloc[2]
        key = row.iloc[0]
        if pd.notnull(value):
            if key != value:
                rename_column[key] = value

    for sheet in dfs_dict.keys():

        if sheet == 'cost' and not dfs_dict['cost'].empty:
            columns = getattr(net, rename_sheet[sheet]).columns

            for column in columns:
                values = getattr(net, rename_sheet[sheet])[column]

                if column == 'et':
                    values_element = getattr(net, rename_sheet[sheet])['element']

                    filtered_values_et = values[values.isin(['gen', 'ext_grid'])]

                    starting_index = filtered_values_et.index[-1] + 1
                    extension = pd.Series(['load'] * len(net.load), index=range(starting_index, starting_index + len(net.load)))
                    filtered_values_et_extended = pd.concat([filtered_values_et,extension])
                    dfs_dict[sheet]['type'] = filtered_values_et_extended

                    result = filtered_values_et.astype(str) + values_element[filtered_values_et.index].astype(str)
                    extension = pd.Series(['load' + str(i) for i in range(len(net.load))], index=range(starting_index, starting_index + len(net.load)))
                    result = pd.concat([result, extension])

                    dfs_dict[sheet]['name'] = result
                    
                elif column == 'element':
                    pass

                else:
                    ### -1 or 0 for cost sheet?
                    values = values[filtered_values_et.index]
                    extension = pd.Series([-1] * len(net.load), index=range(starting_index, starting_index + len(net.load)))
                    values = pd.concat([values,extension])
                    dfs_dict[sheet][column] = values
        
        elif sheet == 'profiles':
            columns_profile = ['asset'] + ['load{}'.format(i) for i in range(len(net.load))]
            dfs_dict['profiles'] = pd.DataFrame(columns=columns_profile)
            dfs_dict['profiles'].loc[1] = ['field'] + ['max_p_mw'] * len(net.load)

        else:

            try:
                columns = getattr(net, rename_sheet[sheet]).columns
                for column in columns:

                    if column in dfs_dict[sheet].keys():
                        old_values = getattr(net, rename_sheet[sheet])[column]
                        values = rename_element(sheet, column, old_values, net, rename)

                        if column == 'type':    # DEBUG HERE--------------------------------------------------------
                            if sheet == 'lines':
                                dfs_dict['ln_type'][column] = values
                        else:
                            dfs_dict[sheet][column] = values
                            del values

                    elif column in rename_column.values(): #and not values.isna().all ?
                        column_update = next((key for key, value in rename_column.items() if value == column), None)
                        if column_update is not None:
                            old_values = getattr(net, rename_sheet[sheet])[column]
                            values = rename_element(sheet, column_update, old_values, net, rename)
                            dfs_dict[sheet][column_update] = values.values
                            del values

                    # elif (sheet == 'lines' and column in dfs_dict['ln_type'].keys()) or (sheet == 'transformers' and column in dfs_dict['tr_type'].keys()):
                    #     values = getattr(net, rename_sheet[sheet])[column]
                    #     values = rename_element(sheet, column, values, net, rename)  # values = ?

                    #     if values.dtype == bool:
                    #         values = values.astype('object').map({True: 'True', False: 'False'})
                        
                    #     if column in dfs_dict['ln_type'].keys():
                    #         dfs_dict['ln_type'][column] = values
                    #     elif column in dfs_dict['tr_type'].keys():
                    #         dfs_dict['tr_type'][column] = values
                    # elif column == 'std_type':
                    #     values = getattr(net, rename_sheet[sheet])[column]
                    #     if sheet == 'transformers':
                    #         for index in values.index:
                    #             values[index] = 'ttype' + str(index)    # TODO
                    #         dfs_dict['tr_type']['name'] = values
                    #     elif sheet == 'lines':
                    #         if column == 'std_type':
                    #             for index in values.index:
                    #                 values[index] = 'ltype' + str(index)
                    #             dfs_dict['ln_type']['name'] = values
                    #     dfs_dict[sheet]['type'] = values

                    else:
                        pass
                        #print(f"\nSheet: {rename_sheet[sheet]}; Column: {column} is NOT used in template sheet")

                if sheet == 'generators':
                    new_dict = {'sgen':pd.DataFrame()}
                    sgen = getattr(net, 'sgen')
                    gen = getattr(net, 'gen')
                    ext_grid = getattr(net, 'ext_grid')

                    for column in dfs_dict[sheet]:
                        if column in sgen.columns and not column == 'type':
                            series_values = sgen[column].reset_index(drop=True)
                            
                            if series_values.dtype == bool:
                                series_values = series_values.astype('object').map({True: 'True', False: 'False'})

                            if column == 'name':
                                for index in series_values.index:
                                    series_values[index] = 'sgen' + str(index)

                        elif column in rename_column.keys():
                            series_values = sgen[rename_column[column]].reset_index(drop=True)
                            bus_list = []
                            for index in series_values.values:
                                bus_list.append(dfs_dict['nodes']['name'].iloc[index])
                                series_values = pd.Series(bus_list, name = column)

                        elif column == 'vm_pu':
                            series_values = pd.Series(dtype='float64')    

                            for value in sgen['bus'].values:
                                at_index = gen['bus'][gen['bus'] == value].index
                                
                                if len(at_index) == 0:
                                    at_index = ext_grid['bus'][ext_grid['bus'] == value].index
                                    series_values = pd.concat([series_values, ext_grid['vm_pu'][at_index]])

                                else:
                                    series_values = pd.concat([series_values, gen['vm_pu'][at_index]])

                            series_values = series_values.reset_index(drop=True)
                            series_values.name = column

                        elif column == 'slack' or column == 'slack_weight':
                            if column == 'slack':
                                series_values = pd.Series(['False'] * len(sgen['bus']), dtype='object')
                            if column == 'slack_weight':
                                series_values = pd.Series([0] * len(sgen['bus']), dtype='object')
                            series_values.name = column

                        else:
                            series_values = pd.Series([] * len(sgen['bus']), dtype='object')
                            series_values.name = column
                        
                        new_dict['sgen'] = pd.concat([new_dict['sgen'], series_values], axis=1)

                    dfs_dict[sheet] = pd.concat([dfs_dict[sheet], new_dict['sgen']])
            except:
                # dfs_dict[sheet]
                print(f"[{sheet},{column}]")

    if net.bus_geodata.index.max() == net.bus.index.max():
        net.bus_geodata = net.bus_geodata.sort_index()
        dfs_dict['nodes']['longitude'] = net.bus_geodata.x
        dfs_dict['nodes']['latitude'] = net.bus_geodata.y

    dfs_dict['network']['sn_mva'] = pd.Series(net.sn_mva)
    dfs_dict['network']['f_hz'] = pd.Series(net.f_hz)
    dfs_dict['network']['name'] = pd.Series(net.name)

    wb = Workbook()
    wb.remove(wb['Sheet'])

    for sheet_name, df in dfs_dict.items():
        ws = wb.create_sheet(sheet_name)
        for row in dataframe_to_rows(df, index=False, header=True):
            ws.append(row)
        style_formatting(ws)

    wb.save('network.xlsx')
    print('\n- network.xlsx created successfully! - ')