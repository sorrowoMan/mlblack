# Dashboard Protocol

这份协议用于约束 `nsgablack` 与 `mlblack` 的 catalog dashboard。
目标不是强行共享所有页面代码，而是共享同一套：

- 查询面契约
- URL / session 状态契约
- 交互语义契约
- 回归验证清单

这样两边即使 kind / field 模型不同，也能长期保持同一层产品语义。

## 1. 产品目标

catalog dashboard 不是示例脚本，而是正式查询页。
最小产品语义：

- 左侧：facet filters
- 中间：可点击结果表格
- 右侧：detail / relations / source 三段详情
- 顶部：scope / profile / kind / query 主控件
- 顶部控件区：统一拆成 `primary` / `secondary` 两排 control row
- URL：可 deep-link 到当前页面状态

## 2. 查询层契约

UI 层只依赖下列统一查询面：

- `catalog_source_info(...)`
- `catalog_summary(...)`
- `catalog_schema(...)`
- `list_entries(...)`
- `search_entries(...)`
- `show_entry(...)`
- `catalog_neighbors(...)`
- `catalog_facets(...)`
- `catalog_ui_snapshot(...)`

不应把 registry / DB / project loader 细节散落进页面代码。

## 3. URL 契约

### 3.1 核心参数

所有 dashboard 都应支持下列 query params：

- `profile`
- `scope`
- `kind`
- `query`
- `selected`
- `sort_by`
- `sort_dir`
- `detail_tab`
- `open_relations`
- `column_mode`
- `page_size`
- `results_collapse`

### 3.2 scope 扩展参数

按需要支持：

- `project_path`
- `include_global`

### 3.3 framework 可选扩展参数

不同框架可增加自己的公共参数，但必须保持含义稳定：

- `field`：例如 `nsgablack` 的搜索范围
- `db_path`：例如 `mlblack` 的 SQL catalog 连接目标
- `source_mode`：例如 `mlblack` 的 registry / DB 读取模式

### 3.4 facet filters

facet filters 统一序列化为：

- `f_<field>=value1,value2`

例如：

- `f_tags=local_search,demo`
- `f_family=neural`

### 3.5 参数取值约束

- `sort_by`: `default | title | key | kind`
- `sort_dir`: `asc | desc`
- `detail_tab`: `overview | relations | source`
- `open_relations`: 逗号分隔的 relation group 名称
- `column_mode`: `compact | standard | full`
- `page_size`: 正整数
- `results_collapse`: `expanded | collapsed`

### 3.6 URL 规则

- deep-link 必须恢复“筛选条件 + 排序 + 当前 tab + 展开 relation 分组 + 选中项 + 结果表格状态”
- `selected=__none__` 这类内部哨兵值不应写进 URL
- 没有值的参数应省略，不应写空字符串噪音

### 3.7 CLI 初始状态

`python -m ... catalog ui` 的正式入口也必须能注入结果区初始状态：

- `--column-mode`
- `--page-size`
- `--results-collapse`

要求：

- CLI 传入的初始值必须在首屏 UI 生效
- 若 URL 同时提供对应 query params，URL 优先级高于 CLI 默认值
- CLI 与 URL 的字段名语义必须保持一致

## 4. Session State 契约

### 4.1 核心页面状态

建议统一使用这些 key：

- `catalog_ui_profile`
- `catalog_ui_scope`
- `catalog_ui_kind`
- `catalog_ui_query`
- `catalog_ui_selected`
- `catalog_ui_project_path`
- `catalog_ui_include_global`

### 4.2 facet state

facet filters 使用：

- `catalog_ui::facet::<scope>::<kind>::<field>`

### 4.3 view state

可交互视图状态使用：

- `catalog_ui::view::<scope>::<kind>::sort_by`
- `catalog_ui::view::<scope>::<kind>::sort_dir`
- `catalog_ui::view::<scope>::<kind>::detail_tab`
- `catalog_ui::view::<scope>::<kind>::open_relations`
- `catalog_ui::view::<scope>::<kind>::column_mode`
- `catalog_ui::view::<scope>::<kind>::page_size`
- `catalog_ui::view::<scope>::<kind>::results_collapse`

### 4.4 navigation state

跳转栈使用：

- `catalog_ui_navigation_stack`

## 5. 交互语义契约

### 5.1 结果表格

结果区必须是主表格，而不是按钮列表。

要求：

- 单击行即可选中
- 选中项联动右侧 detail
- deep-link 能恢复当前选中项
- 当前列显示方案必须是正式状态
- 当前 page size 必须是正式状态
- 当前结果折叠方式必须是正式状态

### 5.2 结果展示状态

结果区的展示方式必须可序列化、可恢复。

要求：

- `column_mode` 必须进入 URL，并能从 URL 恢复
- `page_size` 必须进入 URL，并能从 URL 恢复
- `results_collapse` 必须进入 URL，并能从 URL 恢复
- `page_size` 表示当前结果表格的可见窗口大小，而不是后台查询上限
- 复制 deep-link 后重新打开，结果区的列方案、显示条数、折叠状态必须完全回位

### 5.3 排序

排序是正式状态，不是前端偶发行为。

要求：

- 当前排序必须进入 URL
- 当前排序必须能从 URL 恢复
- `default` 表示保留 catalog 原始顺序

### 5.4 Detail Tabs

右侧 detail 至少拆成三段稳定语义：

- `overview`
- `relations`
- `source`

要求：

- 当前 tab 必须进入 URL
- 当前 tab 必须能从 URL 恢复

### 5.5 Relation Groups

relation group 的展开状态必须是显式状态。

要求：

- 页面必须有可序列化的 relation group 选择面
- 展开状态必须进入 URL
- 展开状态必须能从 URL 恢复

### 5.6 Hidden Selection

若当前选中项被筛选隐藏：

- 不应立即丢失选中态
- 右侧仍应能展示详情
- 页面应明确提示“当前选中项已被筛选隐藏”
- 应提供“显示它”或“清除选中”动作

### 5.7 Control Layout Protocol

顶部控件区不是随手摆的 `st.columns(...)`，而是正式协议的一部分。

要求：

- 页面必须稳定存在 `primary` / `secondary` 两排 control row
- control row id 必须稳定，可供 E2E 与样式协议复用
- control slot 的列宽、label、placeholder、caption、help 骨架必须长期对齐
- 框架可以有各自字段，但不应破坏“第一排主查询、第二排数据源 / 项目 / 补充控制”的布局语义

### 5.8 Button / Rerun Semantics

交互按钮默认应走 callback 改状态，而不是“按钮分支里再显式手动 rerun”。

要求：

- 选中切换、关系跳转、跳转栈返回、清空筛选这类动作优先使用 `on_click`
- 除非确实需要二段状态推进，否则不要在按钮分支里额外调用 `rerun`
- query trace 回归中，一次“上一项 / 下一项”切换应只保留必要 miss
- 当前基线预算：单次选中切换的 loader delta 应控制在 `<= 8`

### 5.7 Relation Jump

从 relation 区跳转时：

- 必须更新 `selected`
- 必须更新跳转栈
- 必须更新 URL

## 6. 回归要求

每次改 dashboard，至少验证：

- framework scope list/search/show
- project scope list/search/show
- deep-link 基础参数恢复
- facet filter 参数恢复
- sort / tab / relation-group / result-layout 参数恢复
- deep-link roundtrip 后 UI snapshot 状态一致
- table selection helper
- hidden selection 行为
- dashboard script import mode

## 7. 框架自由度

协议层允许不同框架保留自己的 kind / field 体系：

- `mlblack`: `family / preset / head / component / provider / plugin`
- `nsgablack`: `adapter / plugin / bias / representation / suite / tool / doc / example`

但它们必须共享同一层 dashboard protocol，而不是长成两套完全不同的产品语义。
