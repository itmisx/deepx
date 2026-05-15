package tui

import (
	"deepx/agent"
	"deepx/session"
	"deepx/skill"
	"deepx/tools"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"charm.land/bubbles/v2/spinner"
	"charm.land/bubbles/v2/textinput"
	"charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"
	"charm.land/glamour/v2"
	"github.com/charmbracelet/x/ansi"
)

// imagePlaceholderRe 匹配输入框里 [Image #N] 形式的图片占位符。
var imagePlaceholderRe = regexp.MustCompile(`\[Image #(\d+)\]`)

type model struct {
	width  int
	height int

	chatViewport viewport.Model
	chatContent  *strings.Builder

	input textinput.Model

	// 整套连接配置 (per-role BaseURL/Model/APIKey),从 ~/.deepx/model.yaml 读取后传入。
	// 旧的 apiKey/baseURL 单独字段已合并到 models 里,通过 models.Flash.XX / models.Pro.XX 访问。
	models agent.ModelConfig

	// 配置 modal 状态:
	//   - showSetup       = true 时 View() 弹模态,Update() 路由按键给 setupInput
	//   - setupRequired   = true 表示无有效 yaml(首次启动);不允许 Esc 关闭,Ctrl+C 才退出
	//   - setupInput      = api key 输入框
	//   - setupErr        = 错误回显(保存失败 / 输入为空等)
	showSetup     bool
	setupRequired bool
	setupInput    textinput.Model
	setupErr      string

	// activeModelRole 是上一轮 / 当前流的实际生效角色 ("flash" / "pro")。
	// 每条新用户消息默认从 flash 起手;agent.ModelSwitchMsg 到达时更新为 pro。
	activeModelRole string
	activeModelID   string

	mode    agent.AgentMode
	history []agent.ChatMessage

	streamCh <-chan tea.Msg

	status    string
	streaming bool
	tokens    int

	currentReply *strings.Builder

	// 待发送的图片已落盘文件路径,Ctrl+V 累加,enter 后清空。
	// 不在内存里囤 PNG 字节是为了:(a) 让 img_ocr 工具能凭路径读;
	// (b) 历史 ChatMessage 之后再次发回 API 时不带巨大的 base64。
	attachedImagePaths []string

	// 当前活跃规划。nil = 无规划。pro 调用 create_plan 时初始化,
	// update_task_status 通过 TaskStatusMsg 增量更新。每次新用户消息发起前清空。
	plan *planState

	// 鼠标 chat 矩形选区。selecting=true 表示左键在 chat 区按下后还没松开;
	// selAnchor / selEnd 是选区两端 (cellPos: 显示列 + wrapped 行号)。
	// 松开左键时:抠出选区内文本写到系统剪贴板,然后 selecting=false 高亮自动消失。
	selecting bool
	selAnchor cellPos
	selEnd    cellPos

	// 输入框全选态(Ctrl+A 触发)。true 时输入框 value 整段反色显示,
	// 下一次按键消费"全选"语义:输入字符 / 删除键 → 清空 value 后再 process;
	// Esc / 方向键 → 仅取消选择,不动 value;
	// Ctrl+A → 维持。
	inputAllSelected bool

	// 命令 palette 选中索引。是否打开不存字段 — 由 filterSlashCommands(input value)
	// 实时计算,避免状态同步问题。idx 越界时(value 一变 matches 变短)由消费方 clamp。
	commandPaletteIdx int

	// lastInputClickAt 记录最近一次落在输入框那一行的左键 click 时间戳。
	// 用来手动检测双击:两次 click 间隔 < 400ms 即视为双击,切换 inputAllSelected。
	// bubbletea v2 的 MouseClickMsg 不带 Clicks 计数,只能自己算。
	lastInputClickAt time.Time

	// 思考动画。streaming=true 且当前没在接收 content tokens 时,在 chat 末尾追加 spinner 帧。
	// reasoning_content / tool_call 阶段都视为"思考中",content token 一到就停;
	// 下一轮工具结果或新 reasoning 到来时再次开启,实现"多轮思考折叠成一个动画"。
	spinner  spinner.Model
	thinking bool

	// scrollbarDragging: 左键在滚动条列按下且未松开。该状态下 motion 事件 → SetYOffset。
	scrollbarDragging bool

	// markdown 实时渲染:每次 refresh 都会用 glamour 重渲整段 chatContent。
	// glamour.NewTermRenderer 是有状态的(配 word wrap 宽度),所以按 width 缓存,
	// width 没变就复用同一个 renderer,避免每帧重建。
	mdRenderer      *glamour.TermRenderer
	mdRendererWidth int

	// session 是当前 workspace 的持久化句柄。启动时建/打开 ~/.deepx/sessions/{sid}/,
	// 写时机:user enter 后 + assistant 流结束(StreamDoneMsg)时各 append 一行。
	// Memory 工具通过 tools.SetMemorySession 拿到同一句柄,扫 jsonl 命中关键词。
	session *session.Manager

	// skill 系统:启动时扫 ~/.deepx/skills/ + session.SkillsDir() 两个目录的 SKILL.md。
	//   - skillLoader 注入 tools.LoadSkill,LLM 调工具时按名取正文
	//   - skillCatalog 在 system prompt 里挂"name + description"摘要,引导 LLM 决定要不要 LoadSkill
	// 进程内不刷新 — 用户增删 skill 后重启 deepx。
	skillLoader  *skill.Loader
	skillCatalog string

	// 右栏仪表盘字段
	workspace       string        // os.Getwd() at startup,展示当前工作目录
	turnStartedAt   time.Time     // 本轮 Enter 时刻,用于实时计算 elapsed
	turnElapsed     time.Duration // 上一轮总耗时,streaming=false 时显示这个
	turnInputChars  int           // 本轮 user 发送时的 history 总字符数(快照)
	turnOutputChars int           // 本轮 assistant content 累计字符数(只算 content,跳过 reasoning)
}

func initialModel(models agent.ModelConfig, needsSetup bool) model {
	vp := viewport.New()

	sp := spinner.New()
	sp.Spinner = spinner.MiniDot

	ti := textinput.New()
	ti.Placeholder = "Type a message... (Enter to send, Ctrl+Shift+A to select all, Ctrl+C to cancel/quit)"
	ti.CharLimit = 4000

	si := textinput.New()
	si.Placeholder = "sk-..."
	si.CharLimit = 256
	si.SetWidth(50)

	// 起手角色 = flash (若 flash 未配置则退化到 pro)
	role := "flash"
	activeID := models.Flash.Model
	if activeID == "" {
		role = "pro"
		activeID = models.Pro.Model
	}

	wd, _ := os.Getwd()

	// 当前 workspace 的会话持久化。建/打开 ~/.deepx/sessions/{sha1(wd)[:16]}/。
	// 失败(权限/磁盘满)不致命 —— sess=nil 时 appendChat 跳过持久化,Memory 工具返回禁用提示。
	sess, sessErr := session.New(wd)
	if sessErr == nil {
		tools.SetMemorySession(sess)
	}

	// Skill 加载器:多来源发现(兼容 Claude Code / opencode / cursor / agents 生态)。
	// workspace 优先于 global,同名覆盖。任一目录不存在不报错。
	// 用户没建任何 skill 时 catalog 为空,system prompt 不挂"Available Skills"段。
	home, _ := os.UserHomeDir()
	workspaceSkillDirs := []string{
		filepath.Join(wd, ".deepx", "skills"),
	}
	globalSkillDirs := []string{
		filepath.Join(home, ".agents", "skills"),
		filepath.Join(home, ".claude", "skills"),
		filepath.Join(home, ".deepx", "skills"),
	}
	loader := skill.New(workspaceSkillDirs, globalSkillDirs)
	tools.SetSkillLoader(loader)
	skillCatalog := buildSkillCatalog(loader)

	m := model{
		chatContent:     &strings.Builder{},
		currentReply:    &strings.Builder{},
		chatViewport:    vp,
		input:           ti,
		models:          models,
		activeModelRole: role,
		activeModelID:   activeID,
		mode:            agent.AgentMode_Auto,
		status:          "idle",
		spinner:         sp,
		workspace:       wd,
		setupInput:      si,
		session:         sess,
		skillLoader:     loader,
		skillCatalog:    skillCatalog,
	}

	// 默认重加最近 N 个 user→assistant 对。失败/空都没事,新会话起步。
	if sess != nil {
		const resumeTurns = 20
		entries := sess.LoadRecentTurns(resumeTurns)
		for _, e := range entries {
			m.history = append(m.history, agent.ChatMessage{
				Role:    e.Role,
				Content: e.Content,
			})
			// chatContent 用统一的角色前缀回显,看起来跟当前会话续上
			role := "deepx"
			if e.Role == "user" {
				role = "You"
			}
			m.chatContent.WriteString(rolePrefix(role) + e.Content + "\n\n")
		}
		if len(entries) > 0 {
			m.chatContent.WriteString("---\n_(以上为历史对话,共 " +
				strconv.Itoa(len(entries)) + " 条)_\n\n")
		}
	}

	// 首次启动:强制弹 modal,焦点在 setup 输入框
	if needsSetup {
		m.showSetup = true
		m.setupRequired = true
		m.setupInput.Focus()
	} else {
		m.input.Focus()
	}

	// endpoint / 模型 / 模式信息全部移到右栏(rightPanelView 直接读 m.models / m.baseURL),
	// chat 区不再发开场 System 消息,保持干净
	return m
}

func (m model) Init() tea.Cmd {
	return textinput.Blink
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		leftW, vpH := m.layout()
		m.chatViewport.SetWidth(leftW)
		m.chatViewport.SetHeight(vpH)
		m.input.SetWidth(leftW - 4)
		// 窗口尺寸变了 → wrap 重算 → 老 line 号失效,必须清选区
		m.selecting = false
		m.refreshViewport()

	case tea.MouseWheelMsg:
		// modal 期间忽略
		if m.showSetup {
			return m, nil
		}
		// 滚轮: 转给 viewport,顺便取消选区
		if m.selecting {
			m.selecting = false
		}
		var c tea.Cmd
		m.chatViewport, c = m.chatViewport.Update(msg)
		return m, c

	case tea.MouseClickMsg:
		if m.showSetup {
			return m, nil
		}
		if msg.Button != tea.MouseLeft {
			return m, nil
		}
		leftW, vpH := m.layout()
		chatLeft, chatTop := 1, 3
		chatRight := chatLeft + leftW
		chatBottom := chatTop + vpH
		scrollbarX := chatRight
		inputRow := chatTop + vpH // 输入框那一行 (chat 高 vpH,紧贴下面)
		inChat := msg.X >= chatLeft && msg.X < chatRight &&
			msg.Y >= chatTop && msg.Y < chatBottom
		onScrollbar := msg.X == scrollbarX &&
			msg.Y >= chatTop && msg.Y < chatBottom
		inInputRow := msg.Y == inputRow && msg.X >= chatLeft && msg.X <= chatRight

		// 输入框双击切换全选(模拟 GUI 双击全选词,但因为输入框单行,这里直接全选整行)。
		// 第一次 click 记下时间戳;400ms 内第二次 click 在同一行 → toggle inputAllSelected。
		if inInputRow {
			now := time.Now()
			if !m.lastInputClickAt.IsZero() && now.Sub(m.lastInputClickAt) < 400*time.Millisecond {
				if m.input.Value() != "" {
					m.inputAllSelected = !m.inputAllSelected
				}
				m.lastInputClickAt = time.Time{} // 清零,避免三击当成第二次双击
			} else {
				m.lastInputClickAt = now
				// 单击进入输入框区域时,若有 chat 区选区高亮 → 清掉,焦点回输入框
				if m.selecting {
					m.selecting = false
					m.refreshViewport()
				}
			}
			return m, nil
		}

		if onScrollbar {
			m.scrollbarDragging = true
			m.scrollbarSeek(msg.Y, chatTop, vpH)
			m.refreshViewport()
		} else if inChat {
			col := msg.X - chatLeft
			line := m.chatViewport.YOffset() + (msg.Y - chatTop)
			m.selAnchor = cellPos{col: col, line: line}
			m.selEnd = m.selAnchor
			m.selecting = true
			m.refreshViewport()
		} else if m.selecting {
			m.selecting = false
			m.refreshViewport()
		}
		return m, nil

	case tea.MouseMotionMsg:
		if m.showSetup {
			return m, nil
		}
		if msg.Button != tea.MouseLeft {
			return m, nil
		}
		leftW, vpH := m.layout()
		chatLeft, chatTop := 1, 3
		chatRight := chatLeft + leftW
		chatBottom := chatTop + vpH

		if m.scrollbarDragging {
			m.scrollbarSeek(msg.Y, chatTop, vpH)
			m.refreshViewport()
		} else if m.selecting {
			scrolled := false
			if msg.Y < chatTop && !m.chatViewport.AtTop() {
				m.chatViewport.ScrollUp(1)
				scrolled = true
			} else if msg.Y >= chatBottom && !m.chatViewport.AtBottom() {
				m.chatViewport.ScrollDown(1)
				scrolled = true
			}
			cx := msg.X
			if cx < chatLeft {
				cx = chatLeft
			}
			if cx >= chatRight {
				cx = chatRight - 1
			}
			cy := msg.Y
			if cy < chatTop {
				cy = chatTop
			}
			if cy >= chatBottom {
				cy = chatBottom - 1
			}
			col := cx - chatLeft
			line := m.chatViewport.YOffset() + (cy - chatTop)
			if scrolled || m.selEnd.col != col || m.selEnd.line != line {
				m.selEnd = cellPos{col: col, line: line}
				m.refreshViewport()
			}
		}
		return m, nil

	case tea.MouseReleaseMsg:
		if m.showSetup {
			return m, nil
		}
		if msg.Button != tea.MouseLeft {
			return m, nil
		}
		if m.scrollbarDragging {
			m.scrollbarDragging = false
		} else if m.selecting {
			text := m.collectSelectionText()
			if text != "" {
				_ = writeClipboardText(text)
			}
			// 不清 selecting:保留高亮,直到用户点别处 / 滚轮 / 改尺寸 / 开始新选择
		}
		return m, nil

	case tea.PasteMsg:
		// 配置 modal 期间,转发给 setupInput(允许粘贴 API key)
		if m.showSetup {
			var c tea.Cmd
			m.setupInput, c = m.setupInput.Update(msg)
			return m, c
		}
		// 空 paste 内容 = 终端有 paste 事件但 PTY 里没文本 → 大概率剪贴板只有图片,主动读 PNG。
		// 非空 paste 内容 = 普通文本粘贴,放掉让 textinput 自己接(它有 PasteMsg 处理)。
		if msg.Content == "" {
			if m.inputAllSelected {
				m.input.SetValue("")
				m.attachedImagePaths = nil
				m.inputAllSelected = false
			}
			if data, err := readClipboardImage(); err == nil {
				path, ferr := saveAttachedImage(data, len(m.attachedImagePaths)+1)
				if ferr != nil {
					m.appendChat("System", "保存粘贴图片失败: "+ferr.Error())
					return m, nil
				}
				m.attachedImagePaths = append(m.attachedImagePaths, path)
				m.insertImagePlaceholder(len(m.attachedImagePaths))
			}
			return m, nil
		}
		// 文本粘贴:让 textinput 处理
		var c tea.Cmd
		m.input, c = m.input.Update(msg)
		return m, c

	case tea.KeyPressMsg:
		// 配置 modal 处于活动状态时,按键全部路由到 setupInput,绕过主界面所有处理。
		if m.showSetup {
			switch msg.String() {
			case "ctrl+c":
				return m, tea.Quit
			case "enter":
				cmd := m.submitSetup()
				return m, cmd
			case "esc":
				// 首次启动 modal 不允许 Esc 关闭(还没有配置可用);
				// 否则正常关闭返回主界面。
				if m.setupRequired {
					m.setupErr = "需先填入 API key 才能继续 (Ctrl+C 退出 deepx)"
					return m, nil
				}
				m.showSetup = false
				m.setupErr = ""
				m.setupInput.Reset()
				m.setupInput.Blur()
				m.input.Focus()
				return m, nil
			}
			var c tea.Cmd
			m.setupInput, c = m.setupInput.Update(msg)
			return m, c
		}

		// 命令 palette 导航键拦截。palette 由 input value 的 "/" 前缀触发,
		// 这里只在 palette 打开时消费 ↑/↓/Tab,其他键继续往下走交给 textinput。
		if matches := filterSlashCommands(m.input.Value()); len(matches) > 0 {
			// clamp idx 避免 value 缩短后越界
			if m.commandPaletteIdx >= len(matches) {
				m.commandPaletteIdx = len(matches) - 1
			}
			if m.commandPaletteIdx < 0 {
				m.commandPaletteIdx = 0
			}
			switch msg.String() {
			case "up":
				if m.commandPaletteIdx > 0 {
					m.commandPaletteIdx--
				}
				return m, nil
			case "down":
				if m.commandPaletteIdx < len(matches)-1 {
					m.commandPaletteIdx++
				}
				return m, nil
			case "tab":
				// 用选中命令替换 input value;光标移到末尾,palette 仍会因 value 完全等于命令
				// 而匹配它自己一条(只剩 1 项),用户可以再按 Enter 执行,或者继续编辑加参数
				m.input.SetValue(matches[m.commandPaletteIdx].name)
				m.input.SetCursor(len(matches[m.commandPaletteIdx].name))
				m.commandPaletteIdx = 0
				return m, nil
			case "enter":
				// palette 打开时 Enter = 直接执行当前选中命令。
				// 不走原 enter 分支的 input value(可能只是 "/sk" 这种半成品 → handleSlashCommand
				// 会报"未知命令")。先把 value 清掉,再执行完整命令。
				chosen := matches[m.commandPaletteIdx].name
				m.input.SetValue("")
				m.commandPaletteIdx = 0
				m.handleSlashCommand(chosen)
				return m, nil
			}
		} else {
			// palette 没在显示,idx 复位避免下次打开时停在过去位置
			m.commandPaletteIdx = 0
		}

		// Ctrl+A 全选态预处理:按下 Ctrl+A 后,任何其他按键都要先消费"全选"语义。
		// 放在外层 switch 之前,确保所有后续 case 都看不到带 selected 的状态。
		if m.inputAllSelected {
			switch msg.String() {
			case "ctrl+shift+a", "super+shift+a":
				return m, nil // 已全选,继续全选
			case "esc":
				m.inputAllSelected = false
				return m, nil
			case "ctrl+c":
				// 保留原"非流式时退出"语义,不被 selected 改变
			case "left", "right", "home", "end", "up", "down",
				"pgup", "pgdown", "pageup", "pagedown":
				// 方向 / 翻页 → 仅取消选择,光标移动交给后续 textinput.Update
				m.inputAllSelected = false
			case "backspace", "delete", "ctrl+w", "ctrl+u":
				m.input.SetValue("")
				m.attachedImagePaths = nil
				m.inputAllSelected = false
				return m, nil
			case "enter":
				// 全选状态按 Enter:按原 enter 流程走(发送当前 value),同时清除选择标记
				m.inputAllSelected = false
				// 不 return,继续走外层 enter case
			default:
				// 可打印字符或其他键:先清空 value,再让 textinput 处理这次按键(就把新字符填入空 value)
				m.input.SetValue("")
				m.attachedImagePaths = nil
				m.inputAllSelected = false
			}
		}
		switch msg.String() {
		// 输入框全选。两个 key 都注册:
		//   - "ctrl+shift+a"  — 跨平台通用,跟 Claude Code 一致 (issue #52912)
		//   - "super+shift+a" — macOS Cmd+Shift+A,Kitty Keyboard Protocol 透传 (iTerm2 /
		//     kitty / WezTerm / Ghostty 默认未占用此组合,自然送达;Terminal.app 不支持
		//     Kitty Protocol,那边用户用 Ctrl+Shift+A 即可)
		//
		// 不用 Cmd+A / Ctrl+A 是因为前者被 macOS 终端 GUI 拦截做"全选窗口文本"、
		// 字节不进 PTY;后者是 readline LineStart 的默认绑定 — Anthropic 的 Claude Code
		// (issue #14789 not_planned) 也确认 Cmd+A 物理不可达,所有 TUI 都用 Ctrl 组合键。
		case "ctrl+shift+a", "super+shift+a":
			if m.input.Value() != "" {
				m.inputAllSelected = true
			}
			return m, nil
		case "ctrl+c":
			if m.streaming {
				// 流式中 Ctrl+C = "中断当前任务",不退出程序。
				// 把 streamCh 交给 drainAndDiscard,后台 goroutine 自己跑完;
				// UI 立刻回到 idle,用户能继续操作或第二次 Ctrl+C 退出。
				if m.streamCh != nil {
					drainAndDiscard(m.streamCh)
					m.streamCh = nil
				}
				m.streaming = false
				m.thinking = false
				m.status = "idle"
				m.chatContent.WriteString("\n[deepx] 已中断 (后台任务会继续到完成或超时)\n\n")
				m.refreshViewport()
				return m, nil
			}
			return m, tea.Quit
		case "up", "down", "pgup", "pgdown", "pageup", "pagedown", "home", "end", "ctrl+u", "ctrl+d":
			var c tea.Cmd
			m.chatViewport, c = m.chatViewport.Update(msg)
			return m, c
		case "ctrl+v":
			// 先看剪贴板有没有图,有就落盘 + 把 [Image #N] 占位符插到输入框光标处。
			// 没图则继续下落到 textinput,走文本粘贴。
			if data, err := readClipboardImage(); err == nil {
				path, err := saveAttachedImage(data, len(m.attachedImagePaths)+1)
				if err != nil {
					m.appendChat("System", "保存粘贴图片失败: "+err.Error())
					return m, nil
				}
				m.attachedImagePaths = append(m.attachedImagePaths, path)
				m.insertImagePlaceholder(len(m.attachedImagePaths))
				return m, nil
			}
		case "enter":
			if m.streaming {
				return m, nil
			}
			input := strings.TrimSpace(m.input.Value())
			if input == "" && len(m.attachedImagePaths) == 0 {
				return m, nil
			}

			// 斜杠命令(本地处理,不发送给 LLM)
			if strings.HasPrefix(input, "/") {
				m.input.SetValue("")
				m.handleSlashCommand(input)
				return m, nil
			}

			userMsg := m.buildUserMessage(input)
			// 聊天窗口里仍显示用户输入的原文 (含占位符),路径替换只发生在发给 LLM 的消息体中。
			m.appendChat("You", input)
			m.history = append(m.history, userMsg)
			// 持久化用户输入到 session 文件。占位符版本(input)而非含完整路径的 LLM 体 —— 重加时
			// 路径已失效但占位符语义清晰;memory 检索时关键词也按用户原话更直观。
			if m.session != nil {
				_ = m.session.Append("user", input)
			}
			m.attachedImagePaths = nil
			m.input.SetValue("")

			m.status = "thinking"
			m.streaming = true
			m.thinking = true
			m.currentReply.Reset()

			// 仪表盘:开计时,快照本轮输入字符数(用 history 总长近似 = 上下文输入),输出归零
			m.turnStartedAt = time.Now()
			m.turnInputChars = sumHistoryChars(m.history)
			m.turnOutputChars = 0

			m.chatContent.WriteString(deepxPrefix)
			m.refreshViewport()
			// 启动 spinner ticking
			cmds = append(cmds, m.spinner.Tick)

			// 每次新用户消息开始,角色重置回 flash (起手就是默认 flash)。
			// agent.StartStream 内部根据 keyword router 决定本轮真实模型,然后通过 ModelSwitchMsg 通知。
			m.activeModelRole = "flash"
			m.activeModelID = m.models.Flash.Model
			if m.activeModelID == "" {
				m.activeModelRole = "pro"
				m.activeModelID = m.models.Pro.Model
			}
			// 上一轮的 plan 清空,避免残留状态混淆当前任务
			m.plan = nil

			workspace, _ := os.Getwd()
			cmd, ch := agent.StartStream(
				m.models,
				m.history, maxTokens,
				m.mode,
				workspace,
				m.skillCatalog,
			)
			m.streamCh = ch
			cmds = append(cmds, cmd)
			return m, tea.Batch(cmds...)
		}

	case agent.ReasoningTokenMsg:
		// 模型在思考。文字不进 chat,只确保 spinner 在转。
		m.status = "thinking"
		if !m.thinking {
			m.thinking = true
			cmds = append(cmds, m.spinner.Tick)
		}
		return m, tea.Batch(append(cmds, agent.ListenToStream(m.streamCh))...)

	case agent.TokenMsg:
		// 助手正式回复开始,停止 spinner,把文本写进 chat
		m.status = "streaming"
		m.thinking = false
		text := string(msg)
		m.currentReply.WriteString(text)
		m.chatContent.WriteString(text)
		m.tokens += len([]rune(text))
		m.turnOutputChars += len([]rune(text))
		m.refreshViewport()
		return m, agent.ListenToStream(m.streamCh)

	case agent.ToolCallStartMsg:
		// 工具调用 = 一次"动作",紧凑单行展示:<icon> Name (主参数)
		m.status = "tool"
		line := formatToolCallLine(msg.Name, msg.Args)
		m.chatContent.WriteString("\n" + line + "\n")
		m.refreshViewport()
		// 工具执行期间继续转 spinner,等结果回来后才可能切到 content
		if !m.thinking {
			m.thinking = true
			cmds = append(cmds, m.spinner.Tick)
		}
		return m, tea.Batch(append(cmds, agent.ListenToStream(m.streamCh))...)

	case agent.ToolCallResultMsg:
		// 结果不再原样打印长输出。失败时短提示一行;成功默默吞掉(LLM 会用结果接着干)。
		// 这样多轮工具调用看到的就是清爽的工具列表,不被几百行输出淹没。
		if !msg.Success {
			out := msg.Output
			if len(out) > 200 {
				out = out[:200] + "…"
			}
			m.chatContent.WriteString("  ✗ " + msg.Name + " 失败: " + out + "\n")
		}
		m.currentReply.Reset()
		m.refreshViewport()
		// 工具结果后,LLM 还要继续思考(或最终回复),保持 spinner 状态
		if !m.thinking {
			m.thinking = true
			cmds = append(cmds, m.spinner.Tick)
		}
		return m, tea.Batch(append(cmds, agent.ListenToStream(m.streamCh))...)

	case spinner.TickMsg:
		// spinner 自驱动:每次 tick 来都更新 frame,只要还在 thinking 就继续 Tick
		var c tea.Cmd
		m.spinner, c = m.spinner.Update(msg)
		if m.thinking {
			cmds = append(cmds, c)
		}
		// thinking 状态下需要重渲染让 spinner 帧切换可见
		m.refreshViewport()
		return m, tea.Batch(cmds...)

	case agent.HistoryUpdateMsg:
		m.history = msg.History
		return m, agent.ListenToStream(m.streamCh)

	case agent.ModelSwitchMsg:
		m.activeModelRole = msg.Role
		m.activeModelID = msg.ModelID
		if msg.Reason != "" {
			// 升级类的切换在聊天流里留一行可见痕迹,便于用户察觉为什么变贵了
			m.chatContent.WriteString(fmt.Sprintf("\n[已升级到 %s 模型] 原因: %s\n", msg.Role, msg.Reason))
			m.refreshViewport()
		}
		return m, agent.ListenToStream(m.streamCh)

	case agent.PlanCreatedMsg:
		m.plan = &planState{items: msg.Plans}
		// 不写 chatContent — plan 通过 refreshViewport 的 live overlay 实时渲染,
		// 每次 TaskStatusMsg 自然刷新出新 checkbox。
		// 流结束时(StreamDoneMsg)再固化一次进 chatContent 留作历史。
		m.refreshViewport()
		return m, agent.ListenToStream(m.streamCh)

	case agent.TaskStatusMsg:
		// 只有 plan 已就位才有意义;否则可能是模型乱调,丢弃
		if m.plan != nil {
			m.plan.apply(msg)
			m.refreshViewport() // 触发 live overlay 重渲染,checkbox 立刻更新
		}
		return m, agent.ListenToStream(m.streamCh)

	case agent.StreamDoneMsg:
		// 流结束时把当前 plan 最终状态固化进 chatContent,这样滚回历史还能看到。
		// 在写 "\n\n" 之前固化,plan 留在本轮回复结尾。
		if m.plan != nil {
			m.chatContent.WriteString("\n" + renderPlanForChat(m.plan))
		}
		m.chatContent.WriteString("\n\n")
		// 持久化助手最终回复。currentReply 在流式过程中累加,这里一次性落盘。
		// 注意只存"主对话内容" —— tool_call / tool_result / reasoning 都不进 session 文件。
		if m.session != nil {
			final := m.currentReply.String()
			if strings.TrimSpace(final) != "" {
				_ = m.session.Append("assistant", final)
			}
		}
		m.status = "idle"
		m.streaming = false
		m.thinking = false
		m.streamCh = nil
		m.turnElapsed = time.Since(m.turnStartedAt)
		m.refreshViewport()
		return m, nil

	case agent.StreamErrMsg:
		m.chatContent.WriteString("\n[Error: " + msg.Err.Error() + "]\n\n")
		m.status = "error"
		m.streaming = false
		m.thinking = false
		m.streamCh = nil
		m.turnElapsed = time.Since(m.turnStartedAt)
		m.refreshViewport()
		return m, nil
	}

	var inputCmd tea.Cmd
	m.input, inputCmd = m.input.Update(msg)
	cmds = append(cmds, inputCmd)

	return m, tea.Batch(cmds...)
}

// buildUserMessage 把输入文本中的 [Image #N] 占位符替换成已落盘的图片绝对路径,
// 让不支持多模态输入的 LLM (例如 DeepSeek) 也能拿到图片信息——
// 模型可通过 img_ocr 工具按路径识别图片内容。
//
// 设计:图片在 Ctrl+V 时就落盘到 ~/.deepx/ocr/cache/<ts>-<N>.png,
// 这里只做"占位符 → 路径"的文本替换,不再产出多模态 ContentParts。
// 用户手动删掉的占位符对应的图片不会出现在最终消息里。
func (m model) buildUserMessage(text string) agent.ChatMessage {
	if len(m.attachedImagePaths) == 0 {
		return agent.ChatMessage{Role: "user", Content: text}
	}
	replaced := imagePlaceholderRe.ReplaceAllStringFunc(text, func(match string) string {
		// match 形如 "[Image #1]";提取数字索引
		sub := imagePlaceholderRe.FindStringSubmatch(match)
		if len(sub) < 2 {
			return match
		}
		idx, _ := strconv.Atoi(sub[1])
		if idx < 1 || idx > len(m.attachedImagePaths) {
			return match
		}
		return m.attachedImagePaths[idx-1]
	})
	return agent.ChatMessage{Role: "user", Content: replaced}
}

// saveAttachedImage 把刚粘贴的 PNG 字节写到 ~/.deepx/ocr/cache/ 下的新文件,
// 返回该文件的绝对路径。失败时返回空串 + 错误,调用方决定怎么提示用户。
func saveAttachedImage(data []byte, index int) (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		home = os.TempDir()
	}
	cacheDir := filepath.Join(home, ".deepx", "ocr", "cache")
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return "", err
	}
	// 时间戳精确到纳秒,避免连续粘贴的文件名冲突
	name := fmt.Sprintf("%s-%d.png", time.Now().Format("20060102-150405.000000000"), index)
	path := filepath.Join(cacheDir, name)
	if err := os.WriteFile(path, data, 0644); err != nil {
		return "", err
	}
	return path, nil
}

// insertImagePlaceholder 在输入框当前光标位置插入 [Image #n],
// 必要时在前后补空格,避免和相邻字符黏在一起。
func (m *model) insertImagePlaceholder(n int) {
	placeholder := fmt.Sprintf("[Image #%d]", n)
	cur := []rune(m.input.Value())
	pos := m.input.Position()
	if pos < 0 || pos > len(cur) {
		pos = len(cur)
	}
	prefix := ""
	if pos > 0 && cur[pos-1] != ' ' {
		prefix = " "
	}
	suffix := ""
	if pos < len(cur) && cur[pos] != ' ' {
		suffix = " "
	}
	ins := []rune(prefix + placeholder + suffix)
	m.input.SetValue(string(cur[:pos]) + string(ins) + string(cur[pos:]))
	m.input.SetCursor(pos + len(ins))
}

// rolePrefix 把内部 role 名映射成 chatContent 里要写入的 markdown 前缀。
// glamour 渲染时 **...** 会变粗体,emoji 给视觉锚点,跨终端比 ANSI 着色稳。
// 未知 role 走兜底 "**role**: ",不会丢字。
func rolePrefix(role string) string {
	switch role {
	case "You":
		return userPrefix
	case "System":
		return systemPrefix
	case "deepx", "Assistant":
		return deepxPrefix
	default:
		return "**" + role + "**: "
	}
}

func (m *model) appendChat(role, text string) {
	m.chatContent.WriteString(rolePrefix(role) + text + "\n\n")
	m.refreshViewport()
}

// handleSlashCommand 处理本地斜杠命令。
func (m *model) handleSlashCommand(input string) {
	cmd := strings.ToLower(strings.TrimSpace(input))
	switch cmd {
	case "/plan":
		m.mode = agent.AgentMode_Plan
		m.appendChat("System", "已切换到 plan 模式(仅只读工具:list_dir / read_file / grep_file / glob_file / file_tree)")
	case "/auto":
		m.mode = agent.AgentMode_Auto
		m.appendChat("System", "已切换到 auto 模式(全部工具可用,包含 write_file / edit_file / run_command)")
	case "/mode":
		m.appendChat("System", fmt.Sprintf("当前模式: %s", m.mode))
	case "/config":
		m.openSetupModal()
	case "/skills":
		m.appendChat("System", m.skillsListMessage())
	case "/help":
		// 走 markdown 列表 — chatContent 经 glamour 渲染,缩进对齐文本会被当 code block 处理,
		// `-` 项目符号更稳。每项格式: "`/cmd` — 说明"。
		m.appendChat("System", "\n**Slash 命令**\n\n"+
			"- `/plan` — 切到只读模式(仅 Read / List / Grep / Glob / Tree / Search / Fetch / Memory)\n"+
			"- `/auto` — 切回全工具模式(默认)\n"+
			"- `/mode` — 显示当前模式\n"+
			"- `/config` — 重新配置 API key (覆盖 `~/.deepx/model.yaml`)\n"+
			"- `/skills` — 列出可用 skill\n"+
			"- `/help` — 帮助\n\n"+
			"**快捷键**\n\n"+
			"- `Enter` — 发送\n"+
			"- `Ctrl+Shift+A` / macOS `Cmd+Shift+A` — 输入框全选\n"+
			"- `Ctrl+V` — 粘贴(含图片)\n"+
			"- `Ctrl+C` — 流式中中断当前回合;空闲时退出")
	default:
		m.appendChat("System", fmt.Sprintf("未知命令: %s (输入 /help 查看)", cmd))
	}
}

// === Skill 辅助 ===

// buildSkillCatalog 把所有可用 skill 拼成给 LLM 看的简短目录。
// 格式:每行 "- <name> (<scope>): <description>"。空 loader / 空目录返回空串。
func buildSkillCatalog(l *skill.Loader) string {
	if l == nil {
		return ""
	}
	metas := l.List()
	if len(metas) == 0 {
		return ""
	}
	var sb strings.Builder
	for _, m := range metas {
		desc := m.Description
		if desc == "" {
			desc = "(无 description)"
		}
		fmt.Fprintf(&sb, "- %s (%s): %s\n", m.Name, m.Scope, desc)
	}
	return sb.String()
}

// skillsListMessage 给 /skills slash 命令的输出。
// 列出当前两个目录里所有 skill,显示 scope / name / description / 路径。
// 没有任何 skill 时给个友好引导。
func (m *model) skillsListMessage() string {
	if m.skillLoader == nil {
		return "skill 系统未启用"
	}
	metas := m.skillLoader.List()
	if len(metas) == 0 {
		var dirsBlock strings.Builder
		for _, d := range m.skillLoader.AllDirs() {
			fmt.Fprintf(&dirsBlock, "  - %s\n", d)
		}
		return "暂无 skill。在以下任一目录创建 <name>/SKILL.md 即可:\n" +
			dirsBlock.String() +
			"\nSKILL.md 格式:\n" +
			"  ---\n" +
			"  name: my-skill\n" +
			"  description: 一句话说明做什么 + 何时用\n" +
			"  ---\n" +
			"  <markdown 正文>"
	}
	var sb strings.Builder
	fmt.Fprintf(&sb, "可用 skill (%d 个):\n\n", len(metas))
	for _, sm := range metas {
		fmt.Fprintf(&sb, "**%s** _(%s)_\n  %s\n  %s\n\n",
			sm.Name, sm.Scope, sm.Description, sm.Path)
	}
	return strings.TrimRight(sb.String(), "\n")
}

// scrollbarSeek 按鼠标 Y 在 chat 高度里的比例,直接 SetYOffset。
// chatTop 是 chat 区首行的屏幕绝对行号,vpH 是 chat 区高度。
func (m *model) scrollbarSeek(mouseY, chatTop, vpH int) {
	total := m.chatViewport.TotalLineCount()
	vis := m.chatViewport.VisibleLineCount()
	if total <= vis || vpH <= 1 {
		return
	}
	rel := mouseY - chatTop
	if rel < 0 {
		rel = 0
	}
	if rel > vpH-1 {
		rel = vpH - 1
	}
	pct := float64(rel) / float64(vpH-1)
	m.chatViewport.SetYOffset(int(pct * float64(total-vis)))
}

// collectSelectionText 从当前 chatContent 抠出选区文本(经过同样的 wrap 再按列裁剪),
// 用于 mouse-release 时写入剪贴板。
func (m *model) collectSelectionText() string {
	w := m.chatViewport.Width()
	if w <= 0 {
		return ""
	}
	content := m.chatContent.String()
	content = ansi.Wrap(content, w, " -")
	return extractSelectionText(content, m.selAnchor, m.selEnd, w)
}

// renderMarkdown 用 glamour 把 chatContent 转成 styled ANSI 输出。
//
// 关键预处理:剥掉 chatContent 里已有的 ANSI 转义(deepxPrefix 这种 lipgloss 着色)。
// glamour 不认 ANSI,会把 \x1b[31m 这种字节当成 literal 字符喷出来,
// 终端就显示 "[31m..." 这种乱码 / VSCode 直接渲染崩。
// 代价:loss deepxPrefix 的蓝色加粗,但 glamour 自己会把 markdown 加粗/标题/列表样式补回来。
func (m *model) renderMarkdown(content string, width int) string {
	if width <= 0 || content == "" {
		return content
	}
	content = ansi.Strip(content)
	if m.mdRenderer == nil || m.mdRendererWidth != width {
		r, err := glamour.NewTermRenderer(
			glamour.WithStandardStyle("dark"),
			glamour.WithWordWrap(width),
		)
		if err != nil {
			return content
		}
		m.mdRenderer = r
		m.mdRendererWidth = width
	}
	out, err := m.mdRenderer.Render(content)
	if err != nil {
		return content
	}
	return out
}

func (m *model) refreshViewport() {
	atBottom := m.chatViewport.AtBottom()
	w := m.chatViewport.Width()

	// markdown 渲染先于其他叠加:把 chatContent (markdown 原文) 转成 styled ANSI 文本。
	// glamour 自身按 width 做 word wrap + 给每个块加 2 格左 margin。
	// 流式中每个 token 都会触发本函数,glamour 重渲整段 — 不完整的 markdown (如未闭合的 **) 当字面渲染。
	// 这就是 ChatGPT 风格的"实时无闪烁"渲染:每帧自洽,新 token 到来后下一帧自然带上。
	//
	// 渲染前先 ensureEmojiSpacing:emoji 后紧跟 CJK / 字母时(LLM 输出 "📁拆大文件" 这种紧凑写法)
	// 终端会按 text presentation 把 emoji 渲染成 1 cell 而非 2 cell,导致行宽估算少 1 → scrollbar
	// 那行左移。在 emoji 后插空格强制 emoji presentation,稳定 2 cell。
	content := m.renderMarkdown(ensureEmojiSpacing(m.chatContent.String()), w)
	// 二次保险:glamour 在表格 cell normalize 时会 trim 掉前面 ensureEmojiSpacing 加的空格,
	// 这里 ANSI-aware 再扫一遍补回 emoji 后的分隔空格(跳过 SGR 序列不破坏 ANSI)。
	content = ensureEmojiSpacingANSI(content)

	// plan 树和 spinner 不是 markdown — 它们是 ANSI 格式化的 widget,glamour 化只会破坏。
	// 在 markdown 渲染后再叠加。
	if m.plan != nil && m.streaming {
		content += "\n" + renderPlanForChat(m.plan)
	}
	if m.thinking {
		content += m.spinner.View() + " thinking..."
	}
	// glamour 自身按 word boundary wrap 到 width,所以不需要再 ansi.Wrap。
	// 但 plan / spinner 那段是后追加的,它们行数不多且本身就短行,wrap 也无意义。
	// 选区高亮在所有渲染叠加之后注入,基于当前可见的 wrapped 行号定位。
	if m.selecting && w > 0 {
		content = applySelectionHighlight(content, m.selAnchor, m.selEnd, w)
	}
	// 不在这里 padLinesToWidth — viewport.SetContent 会按它自己的 width 重新 wrap,
	// 我们 pad 完它再 wrap 一次,padding 就被冲掉了。统一在 view.go 里 viewport.View()
	// 输出之后再 pad,那才是 scrollbar JoinHorizontal 真正用到的宽度。
	m.chatViewport.SetContent(content)
	// 只有用户原本在底部时才自动跟随,否则保持当前阅读位置
	if atBottom {
		m.chatViewport.GotoBottom()
	}
}
